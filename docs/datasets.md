# Datasets

## Local Layout

Datasets are expected under:

```text
data/<dataset_name>/
```

A standard dataset contains:

```text
triplets.txt
qa_<hops>hop.csv
```

For example:

```text
qa_nhop.csv
```

Encoded datasets such as MQuAKE may additionally contain:

```text
node_data.csv
relation_data.csv
```

These mapping files are optional. When they are absent, as in unencoded datasets such as Kinship, entity and relation strings are treated as already human-readable and title mappings are omitted from prompts.

The `data/` directory is a local dataset location and is not part of the checked-in repository tree.

## Datasets Used in the Paper

The paper evaluates the navigation-ready **KINSHIP** and **MQuAKE-ST** resources, including the Single Answer and Multi Answer MQuAKE-ST settings.

Dataset releases, preparation details, and associated THESEUS resources are maintained in the [THESEUS repository](https://github.com/HalcyonSolutions/THESEUS).

For paper reproduction, use the dataset versions distributed through THESEUS rather than reconstructing them from the generic layout description above.

## Dataset Names Used by the Runners

The paper experiment scripts use:

- `kinship`
- `mquake_single`
- `mquake_multi`

See [Reproducibility](reproducibility.md) for the experiment scripts and configured hop/action/context limits.
