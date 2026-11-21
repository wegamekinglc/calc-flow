use std::collections::{BTreeMap, BTreeSet};

use calc_flow::canonical_json;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const PIPELINE_NAME: &str = "checkpoint-v4-golden";
const PIPELINE_FINGERPRINT: &str =
    "1111111111111111111111111111111111111111111111111111111111111111";
const ROOT0_PATH: &str = "lineage/epochs/7/rolling/0/root.cfstate";
const ROOT1_PATH: &str = "lineage/epochs/1/rolling/1/root.cfstate";
const PAGE0_PATH: &str = "lineage/epochs/3/rolling/0/0/3/page.cfstate";

#[derive(Clone, Debug)]
pub struct GoldenBundle {
    manifest: Value,
    recovery: Value,
    partition0: Value,
    partition1: Value,
    page0: Vec<u8>,
}

impl GoldenBundle {
    pub fn valid() -> Self {
        Self {
            manifest: parse(include_str!("../fixtures/checkpoint-v4/manifest-v4.json")),
            recovery: parse(include_str!("../fixtures/checkpoint-v4/recovery-v4.json")),
            partition0: parse(include_str!("../fixtures/checkpoint-v4/partition-0.json")),
            partition1: parse(include_str!("../fixtures/checkpoint-v4/partition-1.json")),
            page0: include_bytes!("../fixtures/checkpoint-v4/page-0.bin").to_vec(),
        }
    }

    pub fn canonical_manifest(&self) -> String {
        canonical(&self.manifest)
    }

    pub fn canonical_recovery(&self) -> String {
        canonical(&self.recovery)
    }

    pub fn canonical_partition0(&self) -> String {
        canonical(&self.partition0)
    }

    pub fn mutated(&self, case: &GoldenCase) -> Self {
        let mut mutated = self.clone();
        match case.target {
            GoldenTarget::Manifest => apply_json_operation(
                &mut mutated.manifest,
                case.operation,
                case.path(),
                case.value.as_ref(),
            ),
            GoldenTarget::Partition0 => apply_json_operation(
                &mut mutated.partition0,
                case.operation,
                case.path(),
                case.value.as_ref(),
            ),
            GoldenTarget::Recovery => apply_json_operation(
                &mut mutated.recovery,
                case.operation,
                case.path(),
                case.value.as_ref(),
            ),
            GoldenTarget::Page0 => {
                let index = case.index.expect("page mutation has an index");
                mutated.page0[index] ^= 0x01;
            }
        }
        if case.repair_root_handle {
            mutated.repair_root0_handle();
        }
        if case.repair_fingerprint {
            repair_manifest_fingerprint(&mut mutated.manifest);
        }
        mutated
    }

    fn repair_root0_handle(&mut self) {
        let bytes = canonical(&self.partition0).into_bytes();
        let root = self
            .manifest
            .pointer_mut("/operators/rolling/partitions/0/root")
            .expect("partition zero root exists");
        root["byte_len"] = Value::from(bytes.len() as u64);
        root["sha256"] = Value::from(hex::encode(Sha256::digest(bytes)));
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GoldenCategory {
    Identity,
    Participant,
    Partition,
    Page,
    Compatibility,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoldenTarget {
    Manifest,
    Partition0,
    Recovery,
    Page0,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoldenOperation {
    Set,
    Remove,
    Duplicate,
    FlipByte,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenCase {
    pub name: String,
    pub category: GoldenCategory,
    target: GoldenTarget,
    operation: GoldenOperation,
    path: Option<String>,
    value: Option<Value>,
    index: Option<usize>,
    repair_root_handle: bool,
    repair_fingerprint: bool,
    pub expected_error: String,
}

impl GoldenCase {
    fn path(&self) -> &str {
        self.path.as_deref().expect("JSON mutation has a path")
    }
}

pub fn golden_cases() -> Vec<GoldenCase> {
    serde_json::from_str(include_str!("../fixtures/checkpoint-v4/cases.json"))
        .expect("golden cases are valid JSON")
}

#[derive(Debug, Eq, PartialEq)]
pub struct RestoredGolden {
    pub source_sequence: u64,
    pub state_bytes: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format_version: u32,
    pipeline_name: String,
    pipeline_fingerprint: String,
    runtime_config_hash: String,
    epoch: u64,
    created_at: String,
    recovery_status: String,
    job_kind: String,
    recovery_fingerprint: String,
    sources: BTreeMap<String, Source>,
    operators: BTreeMap<String, Operator>,
    sinks: BTreeMap<String, Sink>,
    static_inputs: BTreeMap<String, Value>,
    #[serde(rename = "manifest_fingerprint")]
    fingerprint: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    cursor: Option<Value>,
    identity_hash: String,
    next_sequence: u64,
    ended: bool,
    watermark_policy: Value,
    history: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Operator {
    kind: String,
    progress: BTreeMap<String, Value>,
    output_frontier: Option<Value>,
    state_fingerprint: String,
    partitions: Vec<Partition>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sink {
    delivery: Value,
    pre_commit: Option<Value>,
    blocks: Vec<Handle>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Partition {
    key_group: u32,
    root: Handle,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Handle {
    owner: String,
    epoch: u64,
    object_id: String,
    relative_path: String,
    byte_len: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartitionIndex {
    format_version: u32,
    kind: String,
    key_group: u32,
    pages: Vec<PageEntry>,
    next_generations: BTreeMap<u32, u64>,
    free_slots: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PageEntry {
    page_id: GoldenPageId,
    column: String,
    kind: String,
    logical_range: LogicalRange,
    byte_len: u64,
    sha256: String,
    handle: Handle,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Deserialize)]
#[serde(deny_unknown_fields)]
struct GoldenPageId {
    key_group: u32,
    slot: u32,
    generation: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LogicalRange {
    start: u64,
    end: u64,
}

pub fn validate(bundle: &GoldenBundle) -> Result<RestoredGolden, String> {
    validate_version(&bundle.manifest)?;
    validate_manifest_fingerprint(&bundle.manifest)?;
    let manifest: Manifest =
        serde_json::from_value(bundle.manifest.clone()).map_err(|error| json_error(&error))?;
    validate_identity(&manifest, &bundle.recovery)?;
    validate_participants(&manifest)?;
    validate_partition_groups(&manifest)?;
    let state_bytes = restore_partitions(bundle, &manifest)?;
    Ok(RestoredGolden {
        source_sequence: manifest.sources["input"].next_sequence,
        state_bytes,
    })
}

fn validate_partition_groups(manifest: &Manifest) -> Result<(), String> {
    let groups = manifest.operators["rolling"]
        .partitions
        .iter()
        .map(|partition| partition.key_group)
        .collect::<Vec<_>>();
    if groups != [0, 1] {
        return Err("partition groups differ from prepared key groups".into());
    }
    Ok(())
}

fn restore_partitions(bundle: &GoldenBundle, manifest: &Manifest) -> Result<Vec<u8>, String> {
    let mut state_bytes = Vec::new();
    for partition in &manifest.operators["rolling"].partitions {
        validate_partition(bundle, manifest, partition, &mut state_bytes)?;
    }
    Ok(state_bytes)
}

fn validate_version(manifest: &Value) -> Result<(), String> {
    let found = manifest
        .get("format_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| "manifest format_version must be an integer".to_owned())?;
    if found != 4 {
        return Err(format!(
            "UnsupportedVersion {{ expected: 4, found: {found} }}; checkpoint manifest v{found} is not restorable; start a new lineage"
        ));
    }
    Ok(())
}

fn validate_manifest_fingerprint(manifest: &Value) -> Result<(), String> {
    let expected = manifest_fingerprint(manifest)?;
    if manifest.get("manifest_fingerprint").and_then(Value::as_str) != Some(expected.as_str()) {
        return Err("manifest fingerprint does not match its contents".into());
    }
    Ok(())
}

fn validate_identity(manifest: &Manifest, recovery: &Value) -> Result<(), String> {
    validate_pipeline_identity(manifest)?;
    validate_recovery_identity(manifest, recovery)?;
    validate_fingerprint_identity(manifest)?;
    validate_static_identity(manifest)
}

fn validate_pipeline_identity(manifest: &Manifest) -> Result<(), String> {
    if manifest.pipeline_name != PIPELINE_NAME
        || manifest.pipeline_fingerprint != PIPELINE_FINGERPRINT
    {
        return Err("pipeline identity mismatch".into());
    }
    Ok(())
}

fn validate_recovery_identity(manifest: &Manifest, recovery: &Value) -> Result<(), String> {
    let expected_recovery = hex::encode(Sha256::digest(canonical(recovery).as_bytes()));
    if manifest.recovery_fingerprint != expected_recovery {
        return Err("recovery fingerprint mismatch".into());
    }
    Ok(())
}

fn validate_fingerprint_identity(manifest: &Manifest) -> Result<(), String> {
    if manifest.fingerprint.len() != 64
        || !manifest
            .fingerprint
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return Err("manifest fingerprint identity mismatch".into());
    }
    Ok(())
}

fn validate_static_identity(manifest: &Manifest) -> Result<(), String> {
    if manifest.format_version != 4
        || manifest.epoch != 7
        || manifest.created_at != "2026-11-16T00:00:00Z"
        || manifest.runtime_config_hash
            != "3333333333333333333333333333333333333333333333333333333333333333"
        || manifest.recovery_status != "final"
        || manifest.job_kind != "runner"
    {
        return Err("manifest identity mismatch".into());
    }
    Ok(())
}

fn validate_participants(manifest: &Manifest) -> Result<(), String> {
    validate_participant_sets(manifest)?;
    validate_operator_kind(&manifest.operators["rolling"])?;
    validate_source_contract(&manifest.sources["input"])?;
    validate_sink_contract(&manifest.sinks["output"])?;
    validate_operator_contract(&manifest.operators["rolling"])
}

fn validate_participant_sets(manifest: &Manifest) -> Result<(), String> {
    if manifest
        .sources
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>()
        != ["input"]
    {
        return Err("source participants differ".into());
    }
    if manifest
        .operators
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>()
        != ["rolling"]
    {
        return Err("operator participants differ".into());
    }
    if manifest
        .sinks
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>()
        != ["output"]
    {
        return Err("sink participants differ".into());
    }
    if !manifest.static_inputs.is_empty() {
        return Err("static-input participants differ".into());
    }
    Ok(())
}

fn validate_operator_kind(operator: &Operator) -> Result<(), String> {
    if operator.kind != "table" {
        return Err("operator kind mismatch".into());
    }
    Ok(())
}

fn validate_source_contract(source: &Source) -> Result<(), String> {
    if source.cursor.is_none()
        || source.identity_hash
            != "4444444444444444444444444444444444444444444444444444444444444444"
        || source.ended
        || source.watermark_policy.get("kind").and_then(Value::as_str) != Some("source_provided")
        || source.history.is_some()
    {
        return Err("source state contract mismatch".into());
    }
    Ok(())
}

fn validate_sink_contract(sink: &Sink) -> Result<(), String> {
    if sink.delivery.get("kind").and_then(Value::as_str) != Some("ordinary")
        || sink.pre_commit.is_some()
        || !sink.blocks.is_empty()
    {
        return Err("ordinary sink state contract mismatch".into());
    }
    Ok(())
}

fn validate_operator_contract(operator: &Operator) -> Result<(), String> {
    if operator
        .progress
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>()
        != ["input"]
        || operator.output_frontier.is_some()
        || operator.state_fingerprint
            != "5555555555555555555555555555555555555555555555555555555555555555"
    {
        return Err("operator state contract mismatch".into());
    }
    Ok(())
}

fn validate_partition(
    bundle: &GoldenBundle,
    manifest: &Manifest,
    partition: &Partition,
    state_bytes: &mut Vec<u8>,
) -> Result<(), String> {
    let (document, expected_path) = partition_document(bundle, partition.key_group)?;
    validate_partition_root(&partition.root, expected_path, document)?;
    let index = decode_partition_index(document, partition.key_group)?;
    validate_index_order(&index)?;
    validate_partition_metadata(&index)?;
    for page in &index.pages {
        validate_page(bundle, manifest, &index, page, state_bytes)?;
    }
    Ok(())
}

fn partition_document(bundle: &GoldenBundle, group: u32) -> Result<(&Value, &'static str), String> {
    match group {
        0 => Ok((&bundle.partition0, ROOT0_PATH)),
        1 => Ok((&bundle.partition1, ROOT1_PATH)),
        _ => Err("partition groups differ from prepared key groups".into()),
    }
}

fn validate_partition_root(
    root: &Handle,
    expected_path: &str,
    document: &Value,
) -> Result<(), String> {
    if root.relative_path != expected_path {
        return Err("partition root path mismatch".into());
    }
    let bytes = canonical(document).into_bytes();
    if root.byte_len != bytes.len() as u64 || root.sha256 != hex::encode(Sha256::digest(&bytes)) {
        return Err("partition index checksum mismatch".into());
    }
    Ok(())
}

fn decode_partition_index(document: &Value, group: u32) -> Result<PartitionIndex, String> {
    let index: PartitionIndex =
        serde_json::from_value(document.clone()).map_err(|error| json_error(&error))?;
    if index.format_version != 4 || index.kind != "partition_index" {
        return Err("partition index identity mismatch".into());
    }
    if index.key_group != group {
        return Err("partition index key group mismatch".into());
    }
    Ok(index)
}

fn validate_index_order(index: &PartitionIndex) -> Result<(), String> {
    if index
        .pages
        .windows(2)
        .any(|pair| pair[0].page_id >= pair[1].page_id)
    {
        return Err("page entries are not strictly ordered and unique".into());
    }
    let free = index.free_slots.iter().copied().collect::<BTreeSet<_>>();
    if free.len() != index.free_slots.len()
        || index
            .pages
            .iter()
            .any(|page| free.contains(&page.page_id.slot))
    {
        return Err("partition free-slot metadata is inconsistent".into());
    }
    Ok(())
}

fn validate_partition_metadata(index: &PartitionIndex) -> Result<(), String> {
    match index.key_group {
        0 => validate_ring_partition_metadata(index),
        1 => validate_empty_partition_metadata(index),
        _ => Err("partition groups differ from prepared key groups".into()),
    }
}

fn validate_ring_partition_metadata(index: &PartitionIndex) -> Result<(), String> {
    validate_ring_page_inventory(index)?;
    validate_ring_page_contract(&index.pages[0])
}

fn validate_ring_page_inventory(index: &PartitionIndex) -> Result<(), String> {
    if index.pages.len() != 1
        || index.pages[0].page_id
            != (GoldenPageId {
                key_group: 0,
                slot: 0,
                generation: 3,
            })
        || index.next_generations.get(&0) != Some(&4)
        || !index.free_slots.is_empty()
    {
        return Err("page inventory or generation metadata mismatch".into());
    }
    Ok(())
}

fn validate_ring_page_contract(page: &PageEntry) -> Result<(), String> {
    if page.column != "price"
        || page.kind != "ring"
        || page.logical_range.start != 0
        || page.logical_range.end != 4
    {
        return Err("page logical contract mismatch".into());
    }
    Ok(())
}

fn validate_empty_partition_metadata(index: &PartitionIndex) -> Result<(), String> {
    if !index.pages.is_empty()
        || index.next_generations.get(&1) != Some(&2)
        || index.next_generations.get(&3) != Some(&1)
        || index.free_slots != [1, 3]
    {
        return Err("empty partition metadata mismatch".into());
    }
    Ok(())
}

fn validate_page(
    bundle: &GoldenBundle,
    manifest: &Manifest,
    index: &PartitionIndex,
    page: &PageEntry,
    state_bytes: &mut Vec<u8>,
) -> Result<(), String> {
    validate_page_identity(manifest, index, page)?;
    validate_page_payload(bundle, page)?;
    let id = page.page_id;
    state_bytes
        .extend_from_slice(format!("{}:{}:{}:", id.key_group, id.slot, id.generation).as_bytes());
    state_bytes.extend_from_slice(&bundle.page0);
    Ok(())
}

fn validate_page_identity(
    manifest: &Manifest,
    index: &PartitionIndex,
    page: &PageEntry,
) -> Result<(), String> {
    let id = page.page_id;
    if id.key_group != index.key_group {
        return Err("page key group differs from its partition".into());
    }
    let object_id = format!("{}/{}/{}", id.key_group, id.slot, id.generation);
    if page.handle.object_id != object_id || page.handle.owner != "rolling" {
        return Err("page identity differs from handle".into());
    }
    if page.handle.epoch > manifest.epoch || page.handle.byte_len != page.byte_len {
        return Err("page handle epoch or length mismatch".into());
    }
    if page.handle.sha256 != page.sha256 {
        return Err("page digest differs from handle".into());
    }
    if page.handle.relative_path != PAGE0_PATH {
        return Err("page path mismatch".into());
    }
    Ok(())
}

fn validate_page_payload(bundle: &GoldenBundle, page: &PageEntry) -> Result<(), String> {
    if page.byte_len != bundle.page0.len() as u64
        || page.sha256 != hex::encode(Sha256::digest(&bundle.page0))
    {
        let id = page.page_id;
        return Err(format!(
            "page checksum mismatch: rolling/{}/{}/{}",
            id.key_group, id.slot, id.generation
        ));
    }
    Ok(())
}

fn repair_manifest_fingerprint(manifest: &mut Value) {
    let fingerprint = manifest_fingerprint(manifest).expect("canonical mutation input");
    manifest["manifest_fingerprint"] = Value::from(fingerprint);
}

fn manifest_fingerprint(manifest: &Value) -> Result<String, String> {
    let mut unsigned = manifest.clone();
    unsigned
        .as_object_mut()
        .ok_or_else(|| "manifest must be an object".to_owned())?
        .remove("manifest_fingerprint");
    Ok(hex::encode(Sha256::digest(canonical(&unsigned).as_bytes())))
}

fn apply_json_operation(
    document: &mut Value,
    operation: GoldenOperation,
    path: &str,
    value: Option<&Value>,
) {
    match operation {
        GoldenOperation::Set => {
            set_pointer(
                document,
                path,
                value.expect("set mutation has a value").clone(),
            );
        }
        GoldenOperation::Remove => remove_pointer(document, path),
        GoldenOperation::Duplicate => {
            let entry = document
                .pointer(path)
                .expect("duplicate path exists")
                .clone();
            let (parent, _) = path.rsplit_once('/').expect("duplicate has a parent");
            document
                .pointer_mut(parent)
                .and_then(Value::as_array_mut)
                .expect("duplicate parent is an array")
                .push(entry);
        }
        GoldenOperation::FlipByte => panic!("flip_byte targets page bytes, not JSON"),
    }
}

fn set_pointer(document: &mut Value, path: &str, value: Value) {
    if let Some(target) = document.pointer_mut(path) {
        *target = value;
        return;
    }
    let (parent, key) = path.rsplit_once('/').expect("set has a parent");
    document
        .pointer_mut(parent)
        .and_then(Value::as_object_mut)
        .expect("new set parent is an object")
        .insert(key.to_owned(), value);
}

fn remove_pointer(document: &mut Value, path: &str) {
    let (parent, key) = path.rsplit_once('/').expect("remove has a parent");
    let parent = document.pointer_mut(parent).expect("remove parent exists");
    if let Some(object) = parent.as_object_mut() {
        object.remove(key).expect("remove key exists");
    } else if let Some(array) = parent.as_array_mut() {
        array.remove(key.parse::<usize>().expect("array index"));
    } else {
        panic!("remove parent is not a container");
    }
}

fn canonical(value: &Value) -> String {
    canonical_json(value).expect("golden JSON is canonicalizable")
}

fn parse(document: &str) -> Value {
    serde_json::from_str(document).expect("golden fixture is valid JSON")
}

fn json_error(error: &serde_json::Error) -> String {
    format!("golden JSON schema mismatch: {error}")
}
