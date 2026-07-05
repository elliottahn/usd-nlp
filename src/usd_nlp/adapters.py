"""Adapters from conventional NLP pipeline outputs to USD-NLP objects.

The module deliberately has no dependency on spaCy, Stanza, conllu, or any
other external package.  It accepts CoNLL-U text or duck-typed objects that
have already been produced by an upstream NLP engine and converts them into
Scene/Stage/Layer/Prim structures.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .core import LanguageCode, Layer, LayerPurpose, Prim, PrimType, Scene, Stage

try:  # Present in the current release; retained as optional for older builds.
    from .core import Reference
except Exception:  # pragma: no cover
    Reference = None  # type: ignore


def _enum_value(enum_cls: Any, value: Any, fallback: Any) -> Any:
    """Return an enum member from a string, enum member, or fallback."""
    try:
        if isinstance(value, enum_cls):
            return value
    except Exception:
        pass
    if value is None:
        return fallback
    text = str(value)
    for candidate in (text, text.upper(), text.lower()):
        if hasattr(enum_cls, candidate):
            return getattr(enum_cls, candidate)
        try:
            return enum_cls(candidate)
        except Exception:
            pass
    return fallback


def _prim_type(name: str = "CONTEXT") -> Any:
    """Choose a PrimType that works across minor API versions."""
    for candidate in (name, name.upper(), "CONTEXT", "C", "TERM", "T"):
        if hasattr(PrimType, candidate):
            return getattr(PrimType, candidate)
    return next(iter(PrimType))


def _default_layer_purpose() -> Any:
    """Use STRUCTURAL where available; otherwise fall back safely."""
    for candidate in ("STRUCTURAL", "ENTITY_EXTRACTION", "SEGMENTED", "RAW_SOURCE", "L3", "L1", "L0"):
        if hasattr(LayerPurpose, candidate):
            return getattr(LayerPurpose, candidate)
        try:
            return LayerPurpose(candidate)
        except Exception:
            pass
    return next(iter(LayerPurpose))


def _make_prim(prim_id: str, content: str, language: Any, metadata: Dict[str, Any],
               prim_type: Optional[Any] = None) -> Prim:
    """Construct a Prim across old/new USD-NLP constructor names."""
    prim_type = prim_type or _prim_type("CONTEXT")
    try:
        return Prim(prim_id=prim_id, content=content, prim_type=prim_type,
                    language=language, metadata=metadata)
    except TypeError:
        return Prim(id=prim_id, content=content, prim_type=prim_type,
                    language=language, metadata=metadata)


def _prim_id(prim: Prim) -> str:
    return getattr(prim, "prim_id", getattr(prim, "id"))


def _layer_prims(layer: Layer) -> Sequence[Prim]:
    prims = getattr(layer, "prims")
    return list(prims.values()) if isinstance(prims, dict) else list(prims)


def _get_stage(scene: Scene, stage_name: str) -> Optional[Stage]:
    stages = getattr(scene, "stages")
    if isinstance(stages, dict):
        return stages.get(stage_name) or next(iter(stages.values()), None)
    for stage in stages:
        if getattr(stage, "name", None) == stage_name:
            return stage
    return stages[0] if stages else None


def _get_first_layer(stage: Stage) -> Optional[Layer]:
    layers = getattr(stage, "layers")
    if isinstance(layers, dict):
        return next(iter(layers.values()), None)
    return layers[0] if layers else None


def parse_feats(value: Any) -> Dict[str, str]:
    """Parse a CoNLL-U FEATS column into a dictionary."""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if not value or value == "_":
        return {}
    feats: Dict[str, str] = {}
    for item in str(value).split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            feats[k] = v
    return feats


# Backwards-compatible private name used in earlier drafts.
_parse_feats = parse_feats


def _parse_misc(value: str) -> Dict[str, str]:
    if not value or value == "_":
        return {}
    misc: Dict[str, str] = {}
    for item in value.split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            misc[k] = v
        else:
            misc[item] = "true"
    return misc


def _normalise_sent_id(raw: Any, sent_counter: int) -> str:
    text = str(raw).strip() if raw is not None else ""
    if not text or text == "_":
        return f"s{sent_counter}"
    if text.startswith("s"):
        return text
    return f"s{text}"


def _add_dependency_reference(scene: Scene, stage_name: str, prim: Prim,
                              head_prim_id: str, deprel: str) -> None:
    """Record a dependency edge on the Prim and, where supported, on Scene."""
    metadata = getattr(prim, "metadata", {}) or {}
    metadata["head_prim_id"] = head_prim_id
    metadata["deprel"] = deprel
    metadata["dependency_relation"] = deprel
    setattr(prim, "metadata", metadata)

    if hasattr(prim, "add_reference"):
        try:
            prim.add_reference(head_prim_id)  # type: ignore[attr-defined]
        except Exception:
            pass

    if Reference is not None and hasattr(scene, "add_reference"):
        try:
            ref = Reference(source_stage=stage_name, source_prim_id=_prim_id(prim),
                            target_stage=stage_name, target_prim_id=head_prim_id,
                            metadata={"relation": "dependency", "deprel": deprel})
            scene.add_reference(ref)
        except Exception:
            pass


def add_named_entity_span(target: Any, *args: Any, **kwargs: Any) -> Prim:
    """Add a named-entity span Prim referencing covered token Prim identifiers.

    Supported call forms:
      add_named_entity_span(scene, stage_name="doc", token_prim_ids=[...], label="ORG", text="...")
      add_named_entity_span(layer, "span_id", ["token_id"], "ORG")
    """
    if isinstance(target, Layer):
        layer = target
        span_id = str(args[0]) if len(args) > 0 else str(kwargs.get("span_id", "entity_span"))
        token_prim_ids = list(args[1]) if len(args) > 1 else list(kwargs.get("token_prim_ids", []))
        label = str(args[2]) if len(args) > 2 else str(kwargs.get("label", "ENTITY"))
        text = str(args[3]) if len(args) > 3 else str(kwargs.get("text", label))
        language = kwargs.get("language", getattr(layer, "language", LanguageCode.EN))
    else:
        scene = target
        stage_name = str(kwargs.get("stage_name"))
        stage = _get_stage(scene, stage_name)
        if stage is None:
            raise ValueError(f"Stage not found: {stage_name}")
        layer = _get_first_layer(stage)
        if layer is None:
            raise ValueError(f"Stage has no layers: {stage_name}")
        token_prim_ids = list(kwargs.get("token_prim_ids", []))
        label = str(kwargs.get("label", "ENTITY"))
        text = str(kwargs.get("text", label))
        span_id = str(kwargs.get("span_id") or f"{stage_name}:ent{len(_layer_prims(layer)) + 1}")
        language = kwargs.get("language", getattr(layer, "language", LanguageCode.EN))

    lang = _enum_value(LanguageCode, language, getattr(LanguageCode, "EN"))
    metadata: Dict[str, Any] = {
        "annotation_kind": "named_entity",
        "annotation_level": "span",
        "span_type": "named_entity",
        "label": label,
        "token_prim_ids": list(token_prim_ids),
        "source_engine": kwargs.get("source_engine"),
        "start_char": kwargs.get("start_char"),
        "end_char": kwargs.get("end_char"),
        "confidence": kwargs.get("confidence"),
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}
    prim = _make_prim(span_id, text, lang, metadata, _prim_type("CONTEXT"))
    if hasattr(prim, "add_reference"):
        for token_id in token_prim_ids:
            try:
                prim.add_reference(token_id)  # type: ignore[attr-defined]
            except Exception:
                pass
    layer.add_prim(prim)
    return prim


def from_token_records(records: Iterable[Dict[str, Any]], *, scene_name: str = "NLPImport",
                       stage_name: str = "document", language: Any = "en",
                       layer_purpose: Any = None) -> Scene:
    """Convert dictionary-like token records into a USD-NLP Scene.

    Accepted keys include id, text/form, lemma, upos/pos, xpos/tag, feats/morph,
    head, deprel, start_char, end_char, sent_id, and ent_type.
    """
    lang = _enum_value(LanguageCode, language, getattr(LanguageCode, "EN"))
    purpose = _enum_value(LayerPurpose, layer_purpose, _default_layer_purpose())
    scene = Scene(name=scene_name)
    stage = Stage(name=stage_name)
    layer = Layer(purpose=purpose, language=lang)

    token_ids: Dict[Tuple[str, str], str] = {}
    pending_edges: List[Tuple[str, str, str, str]] = []
    roots: List[str] = []

    for index, record in enumerate(records, start=1):
        sent_id = _normalise_sent_id(record.get("sent_id", "s1"), 1)
        tok_id = str(record.get("id", index))
        prim_id = str(record.get("prim_id", f"{stage_name}:{sent_id}:t{tok_id}"))
        form = str(record.get("text", record.get("form", record.get("FORM", ""))))
        head = record.get("head", record.get("HEAD", None))
        deprel = str(record.get("deprel", record.get("DEPREL", "")))

        metadata = {
            "annotation_kind": "token",
            "annotation_level": "token",
            "sent_id": sent_id,
            "sentence_id": sent_id,
            "token_id": tok_id,
            "lemma": record.get("lemma", record.get("LEMMA")),
            "upos": record.get("upos", record.get("pos", record.get("UPOS"))),
            "xpos": record.get("xpos", record.get("tag", record.get("XPOS"))),
            "feats": parse_feats(record.get("feats", record.get("morph", record.get("FEATS", {})))),
            "head": head,
            "deprel": deprel,
            "deps": record.get("deps"),
            "misc": record.get("misc"),
            "start_char": record.get("start_char"),
            "end_char": record.get("end_char"),
            "ent_type": record.get("ent_type"),
            "source_engine": record.get("source_engine"),
        }
        metadata = {k: v for k, v in metadata.items() if v not in (None, "_")}
        prim = _make_prim(prim_id, form, lang, metadata, _prim_type("CONTEXT"))
        layer.add_prim(prim)
        token_ids[(sent_id, tok_id)] = prim_id

        if head not in (None, "", "_", 0, "0"):
            pending_edges.append((prim_id, sent_id, str(head), deprel))
        else:
            prim.metadata["is_dependency_root"] = True
            roots.append(prim_id)

    stage.add_layer(layer)
    if not hasattr(stage, "metadata") or getattr(stage, "metadata", None) is None:
        setattr(stage, "metadata", {})
    stage.metadata.setdefault("dependency_roots", roots)
    stage.metadata.setdefault("source_format", "CoNLL-U")
    scene.add_stage(stage)

    prim_by_id = {_prim_id(p): p for p in _layer_prims(layer)}
    for dep_id, sent_id, head_id, deprel in pending_edges:
        head_prim_id = token_ids.get((sent_id, head_id), f"{stage_name}:{sent_id}:t{head_id}")
        dep = prim_by_id.get(dep_id)
        if dep is not None:
            _add_dependency_reference(scene, stage_name, dep, head_prim_id, deprel)
    return scene


def from_conllu(conllu_text: str, *, scene_name: str = "CoNLLUImport",
                stage_name: str = "document", language: Any = "en",
                layer_purpose: Any = None) -> Scene:
    """Convert CoNLL-U formatted text into a USD-NLP Scene.

    Multi-word token lines (IDs containing '-') and empty nodes (IDs containing
    '.') are skipped, matching standard CoNLL-U processing practice.
    """
    records: List[Dict[str, Any]] = []
    current_sent_id = "s1"
    sent_counter = 1
    for raw_line in conllu_text.splitlines():
        line = raw_line.strip()
        if not line:
            sent_counter += 1
            current_sent_id = f"s{sent_counter}"
            continue
        if line.startswith("#"):
            if line.startswith("# sent_id") and "=" in line:
                current_sent_id = _normalise_sent_id(line.split("=", 1)[1].strip(), sent_counter)
            continue
        parts = line.split("\t")
        if len(parts) != 10:
            continue
        tok_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = parts
        if "-" in tok_id or "." in tok_id:
            continue
        records.append({
            "id": tok_id,
            "text": form,
            "lemma": None if lemma == "_" else lemma,
            "upos": None if upos == "_" else upos,
            "xpos": None if xpos == "_" else xpos,
            "feats": parse_feats(feats),
            "head": head,
            "deprel": None if deprel == "_" else deprel,
            "deps": None if deps == "_" else deps,
            "misc": _parse_misc(misc),
            "sent_id": current_sent_id,
            "source_engine": "conllu",
        })
    return from_token_records(records, scene_name=scene_name, stage_name=stage_name,
                              language=language, layer_purpose=layer_purpose)


# Backward-compatible aliases for documentation variants.
conllu_to_scene = from_conllu


def from_stanza_document(doc: Any, *, scene_name: str = "StanzaImport",
                         stage_name: str = "document", language: Any = "en") -> Scene:
    """Duck-typed converter for a Stanza Document already produced upstream."""
    records: List[Dict[str, Any]] = []
    for sent_index, sentence in enumerate(getattr(doc, "sentences", []), start=1):
        sent_id = _normalise_sent_id(getattr(sentence, "sent_id", f"s{sent_index}"), sent_index)
        for word in getattr(sentence, "words", []):
            records.append({
                "id": getattr(word, "id", len(records) + 1),
                "text": getattr(word, "text", ""),
                "lemma": getattr(word, "lemma", None),
                "upos": getattr(word, "upos", None),
                "xpos": getattr(word, "xpos", None),
                "feats": getattr(word, "feats", None),
                "head": getattr(word, "head", None),
                "deprel": getattr(word, "deprel", None),
                "sent_id": sent_id,
                "source_engine": "stanza",
            })
    return from_token_records(records, scene_name=scene_name,
                              stage_name=stage_name, language=language)


stanza_doc_to_scene = from_stanza_document


def from_spacy_doc(doc: Any, *, scene_name: str = "SpacyImport",
                   stage_name: str = "document", language: Any = "en") -> Scene:
    """Duck-typed converter for a spaCy Doc already produced upstream."""
    records: List[Dict[str, Any]] = []
    for token in doc:
        head = 0 if getattr(token, "head", token) is token else getattr(token.head, "i", 0) + 1
        morph = token.morph.to_dict() if hasattr(token, "morph") and hasattr(token.morph, "to_dict") else {}
        records.append({
            "id": getattr(token, "i", len(records)) + 1,
            "text": getattr(token, "text", ""),
            "lemma": getattr(token, "lemma_", None),
            "upos": getattr(token, "pos_", None),
            "xpos": getattr(token, "tag_", None),
            "feats": morph,
            "head": head,
            "deprel": getattr(token, "dep_", None),
            "ent_type": getattr(token, "ent_type_", None),
            "start_char": getattr(token, "idx", None),
            "end_char": getattr(token, "idx", 0) + len(getattr(token, "text", "")),
            "source_engine": "spacy",
        })
    scene = from_token_records(records, scene_name=scene_name,
                               stage_name=stage_name, language=language)
    for i, ent in enumerate(getattr(doc, "ents", []), start=1):
        token_ids = [f"{stage_name}:s1:t{j + 1}" for j in range(getattr(ent, "start", 0), getattr(ent, "end", 0))]
        add_named_entity_span(scene, stage_name=stage_name, token_prim_ids=token_ids,
                              label=getattr(ent, "label_", "ENTITY"),
                              text=getattr(ent, "text", ""), language=language,
                              source_engine="spacy", start_char=getattr(ent, "start_char", None),
                              end_char=getattr(ent, "end_char", None),
                              span_id=f"{stage_name}:s1:ent{i}")
    return scene


spacy_doc_to_scene = from_spacy_doc
