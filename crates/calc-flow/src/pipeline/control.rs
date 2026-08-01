use std::collections::{BTreeMap, BTreeSet};

use super::{CompiledNode, PortEndpoint};
use crate::{CalcFlowError, runtime::RuntimeEnvelope};

pub(super) enum ControlRouteStatus {
    Supported(ControlRoute),
    Unsupported(UnsupportedControlTopology),
}

pub(super) struct ControlRoute {
    pub(super) origin_step: usize,
    pub(super) steps: Vec<ControlRouteStep>,
}

pub(super) struct ControlRouteStep {
    pub(super) target_node_index: usize,
    pub(super) target: PortEndpoint,
    pub(super) ingress: ControlIngress,
    pub(super) successor_step_indices: Vec<usize>,
}

pub(super) struct PendingControlStep {
    pub(super) step_index: usize,
    pub(super) envelope: RuntimeEnvelope,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum ControlIngress {
    External { input_name: String },
    Edge { source: PortEndpoint },
    UnboundDeclared,
}

struct PotentialControlIngress {
    target: PortEndpoint,
    ingress: ControlIngress,
}

pub(super) struct UnsupportedControlTopology {
    node_id: String,
    ingresses: Vec<PotentialControlIngress>,
}

impl UnsupportedControlTopology {
    pub(super) fn error(&self, input: &str) -> CalcFlowError {
        let labels = self
            .ingresses
            .iter()
            .map(PotentialControlIngress::label)
            .collect::<Vec<_>>();
        CalcFlowError::InvalidArgument {
            field: "control_input".into(),
            message: format!(
                "control dispatch from graph input {input:?} reaches unsupported multi-input node {:?}: {labels:?}",
                self.node_id
            ),
        }
    }
}

impl PotentialControlIngress {
    fn label(&self) -> String {
        let target = format!("{}.{}", self.target.node_id, self.target.port);
        match &self.ingress {
            ControlIngress::External { input_name } => {
                format!("{target} <- graph input {input_name:?}")
            }
            ControlIngress::Edge { source } => {
                format!("{target} <- {}.{}", source.node_id, source.port)
            }
            ControlIngress::UnboundDeclared => format!("{target} <- unbound"),
        }
    }
}

pub(super) fn derive_control_routes(
    nodes: &[CompiledNode],
    external_inputs: &BTreeMap<String, PortEndpoint>,
) -> BTreeMap<String, ControlRouteStatus> {
    let node_indices = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let external_names = external_inputs
        .iter()
        .map(|(name, endpoint)| (endpoint.clone(), name.clone()))
        .collect::<BTreeMap<_, _>>();

    external_inputs
        .iter()
        .map(|(input_name, origin)| {
            let status = derive_control_route(nodes, &node_indices, &external_names, origin);
            (input_name.clone(), status)
        })
        .collect()
}

fn derive_control_route(
    nodes: &[CompiledNode],
    node_indices: &BTreeMap<&str, usize>,
    external_names: &BTreeMap<PortEndpoint, String>,
    origin: &PortEndpoint,
) -> ControlRouteStatus {
    let mut reachable = BTreeSet::from([node_indices[origin.node_id.as_str()]]);
    for (node_index, node) in nodes.iter().enumerate() {
        if !reachable.contains(&node_index) {
            continue;
        }
        for target in node.outbound.values().flatten() {
            reachable.insert(node_indices[target.node_id.as_str()]);
        }
    }

    let potential = reachable
        .iter()
        .map(|node_index| {
            (
                *node_index,
                potential_ingresses(&nodes[*node_index], external_names),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if let Some((node_index, ingresses)) =
        potential.iter().find(|(_, ingresses)| ingresses.len() != 1)
    {
        return ControlRouteStatus::Unsupported(UnsupportedControlTopology {
            node_id: nodes[*node_index].node_id.clone(),
            ingresses: ingresses
                .iter()
                .map(|ingress| PotentialControlIngress {
                    target: ingress.target.clone(),
                    ingress: ingress.ingress.clone(),
                })
                .collect(),
        });
    }

    let step_by_node = reachable
        .iter()
        .enumerate()
        .map(|(step_index, node_index)| (*node_index, step_index))
        .collect::<BTreeMap<_, _>>();
    let steps = reachable
        .iter()
        .map(|node_index| {
            let ingress = &potential[node_index][0];
            let mut successors = nodes[*node_index]
                .outbound
                .values()
                .flatten()
                .map(|target| step_by_node[&node_indices[target.node_id.as_str()]])
                .collect::<Vec<_>>();
            successors.sort_unstable();
            successors.dedup();
            ControlRouteStep {
                target_node_index: *node_index,
                target: ingress.target.clone(),
                ingress: ingress.ingress.clone(),
                successor_step_indices: successors,
            }
        })
        .collect();
    ControlRouteStatus::Supported(ControlRoute {
        origin_step: step_by_node[&node_indices[origin.node_id.as_str()]],
        steps,
    })
}

fn potential_ingresses(
    node: &CompiledNode,
    external_names: &BTreeMap<PortEndpoint, String>,
) -> Vec<PotentialControlIngress> {
    let mut ingresses = node
        .input_ports
        .iter()
        .map(|port| {
            let target = PortEndpoint {
                node_id: node.node_id.clone(),
                port: port.name().into(),
            };
            let ingress = node.inbound.get(port.name()).map_or_else(
                || {
                    external_names.get(&target).map_or(
                        ControlIngress::UnboundDeclared,
                        |input_name| ControlIngress::External {
                            input_name: input_name.clone(),
                        },
                    )
                },
                |source| ControlIngress::Edge {
                    source: source.clone(),
                },
            );
            PotentialControlIngress { target, ingress }
        })
        .collect::<Vec<_>>();
    ingresses.sort_by(|left, right| {
        left.target
            .cmp(&right.target)
            .then_with(|| left.ingress.cmp(&right.ingress))
    });
    ingresses
}
