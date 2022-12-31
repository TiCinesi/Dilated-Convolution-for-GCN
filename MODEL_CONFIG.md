

| layer_type     | Edge Features (i.e. use_edge_features: true ) | edge_agg options |
| -------------  | ----- | --------------    |
| ginconv_paper  | False |                   |
| edge_ginconv   | True  | only 'add'        |

| edge_gineconv  | True  | Not used.         |

| edge_gatconv   | True  | Not used.         |
| gatconv        | False |                   |

| gcnconv        | False |                   |
| edge_gcnconv   | True  | 'add', 'concat'   |
| sageconv       | False |                   |
| edge_sageconv  | True  | 'add', 'concat'   |

| edge_gatv2conv | True  | Not used.         |
| gatv2conv      | False |                   |




| dataset_id | None features | Edge features | Num graphs |      Category  |
| ---------  | ------------  | ------------  | ---------- | -------------- |
| BZR_MD     | yes           |  yes          |    306     | Small molecule | accuracy
| COX2_MD    | yes           |  yes          |    303     | Small molecule | accuracy
| Cuneiform  | yes           |  yes          |    267     | ComputerVision | auc

| ENZYMES    | yes           |  no           |    600     | Bioinformatics | accuracy
| PROTEINS   | yes           |  no           |    1113    | Bioinformatics | accuracy
| DD         | yes           |  no           |    1178    | Bioinformatics | accuracy
| NCI1       | yes           |  no           |    4110    | Bioinformatics | accuracy



| COLLAB     | no            |  no           |    5000    | Social Network | think about what features
| IMDB-BINARY| no            |  no           |    1000    | Social Network | think about what features
| Fingerprint| yes           |  yes          |    2800    | ComputerVision | not working, strange shape issues


