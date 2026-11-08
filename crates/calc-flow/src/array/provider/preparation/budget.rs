use crate::{CalcFlowError, Result};
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
pub(crate) struct Group {
    pub retained: usize,
    pub workspace: usize,
}
#[derive(Clone, Copy)]
pub(crate) struct Alias<'a> {
    pub group: usize,
    pub name: &'a str,
}
pub(crate) struct Consumer<'a> {
    pub node: &'a str,
    pub limit: usize,
    pub aliases: &'a [Alias<'a>],
}

pub(crate) fn admit(groups: &[Group], consumers: &[Consumer<'_>]) -> Result<Vec<usize>> {
    let mut order = (0..consumers.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|index| consumers[*index].node);
    let mut reservations = vec![0; consumers.len()];
    for index in order {
        reservations[index] = admit_one(groups, &consumers[index])?;
    }
    Ok(reservations)
}
fn admit_one(groups: &[Group], consumer: &Consumer<'_>) -> Result<usize> {
    let aliases = distinct(consumer.aliases);
    let reserved = reservation(groups, consumer, &aliases)?;
    for alias in aliases {
        let required = reserved
            .checked_add(group(groups, alias.group)?.workspace)
            .ok_or_else(|| failure(consumer, alias.name, None))?;
        if required > consumer.limit {
            return Err(failure(consumer, alias.name, Some(required)));
        }
    }
    Ok(reserved)
}
fn distinct<'a>(aliases: &'a [Alias<'a>]) -> Vec<&'a Alias<'a>> {
    let mut aliases = aliases.iter().collect::<Vec<_>>();
    aliases.sort_unstable_by_key(|alias| alias.name);
    let mut seen = BTreeSet::new();
    aliases
        .into_iter()
        .filter(|alias| seen.insert(alias.group))
        .collect()
}
fn reservation(groups: &[Group], consumer: &Consumer<'_>, aliases: &[&Alias<'_>]) -> Result<usize> {
    aliases.iter().try_fold(0_usize, |total, alias| {
        total
            .checked_add(group(groups, alias.group)?.retained)
            .ok_or_else(|| failure(consumer, alias.name, None))
    })
}
fn group(groups: &[Group], index: usize) -> Result<Group> {
    groups
        .get(index)
        .copied()
        .ok_or_else(|| CalcFlowError::Internal {
            message: "native static owner group is missing".into(),
        })
}
fn failure(consumer: &Consumer<'_>, alias: &str, required: Option<usize>) -> CalcFlowError {
    let required = required.map_or_else(
        || "a byte count that overflowed usize".into(),
        |bytes| format!("{bytes} bytes"),
    );
    CalcFlowError::InvalidArgument {
        field: format!("static_inputs.{alias}.bytes"),
        message: format!(
            "native static preparation for node '{}' requires {required}, limit is {}",
            consumer.node, consumer.limit
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CalcFlowError;

    #[test]
    fn test_tight_consumer_cannot_borrow_a_wider_consumers_limit() {
        let groups = [Group {
            retained: 100,
            workspace: 4096,
        }];
        let aliases = [Alias {
            group: 0,
            name: "tight.beta",
        }];
        let wider = [Alias {
            group: 0,
            name: "wide.beta",
        }];
        let consumers = [
            Consumer {
                node: "wide",
                limit: 9000,
                aliases: &wider,
            },
            Consumer {
                node: "tight",
                limit: 4195,
                aliases: &aliases,
            },
        ];
        let error = admit(&groups, &consumers).unwrap_err();
        assert!(
            matches!(error,CalcFlowError::InvalidArgument {field,..} if field=="static_inputs.tight.beta.bytes")
        );
    }
    #[test]
    fn test_distinct_owner_reservations_are_summed_before_any_workspace_is_admitted() {
        let groups = [
            Group {
                retained: 100,
                workspace: 20,
            },
            Group {
                retained: 50,
                workspace: 30,
            },
        ];
        let aliases = [
            Alias {
                group: 0,
                name: "calc.a",
            },
            Alias {
                group: 0,
                name: "calc.b",
            },
            Alias {
                group: 1,
                name: "calc.c",
            },
        ];
        let consumers = [Consumer {
            node: "calc",
            limit: 180,
            aliases: &aliases,
        }];
        assert_eq!(admit(&groups, &consumers).unwrap(), [150]);
        let too_small = [Consumer {
            node: "calc",
            limit: 179,
            aliases: &aliases,
        }];
        let error = admit(&groups, &too_small).unwrap_err();
        assert!(
            matches!(error,CalcFlowError::InvalidArgument {field,..} if field=="static_inputs.calc.c.bytes")
        );
    }
    #[test]
    fn test_unrelated_owners_do_not_acquire_an_unrelated_consumers_smaller_limit() {
        let groups = [
            Group {
                retained: 100,
                workspace: 20,
            },
            Group {
                retained: 1000,
                workspace: 100,
            },
        ];
        let small = [Alias {
            group: 0,
            name: "small.beta",
        }];
        let large = [Alias {
            group: 1,
            name: "large.beta",
        }];
        let consumers = [
            Consumer {
                node: "small",
                limit: 120,
                aliases: &small,
            },
            Consumer {
                node: "large",
                limit: 1100,
                aliases: &large,
            },
        ];
        assert_eq!(admit(&groups, &consumers).unwrap(), [100, 1000]);
    }
    #[test]
    fn test_reservation_and_workspace_sums_are_checked_without_overflow_or_default_caps() {
        let groups = [Group {
            retained: usize::MAX,
            workspace: 1,
        }];
        let aliases = [Alias {
            group: 0,
            name: "calc.beta",
        }];
        assert!(
            admit(
                &groups,
                &[Consumer {
                    node: "calc",
                    limit: usize::MAX,
                    aliases: &aliases
                }]
            )
            .is_err()
        );
        assert_eq!(
            admit(
                &[],
                &[Consumer {
                    node: "dynamic",
                    limit: 0,
                    aliases: &[]
                }]
            )
            .unwrap(),
            [0]
        );
    }
}
