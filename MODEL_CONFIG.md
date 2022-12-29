

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
| BZR_MD     | yes           |  yes          |    306     | Small molecule |
| COX2_MD    | yes           |  yes          |    303     | Small molecule |
| Cuneiform  | yes           |  yes          |    267     | ComputerVision |
| Fingerprint| yes           |  yes          |    2800    | ComputerVision |

| ENZYMES    | yes           |  no           |    600     | Bioinformatics |
| PROTEINS   | yes           |  no           |    1113    | Bioinformatics |
| DD         | yes           |  no           |    1178    | Bioinformatics |
| NCI1       | yes           |  no           |    4110    | Bioinformatics |

| COLLAB     | no            |  no           |    5000    | Social Network |
| IMDB-BINARY| no            |  no           |    1000    | Social Network |


