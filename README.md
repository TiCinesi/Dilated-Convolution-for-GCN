# Deep learning project

Nat: 11 8>=
Gio: 5

Errors: 17 (+ 1 perche dataset tu è strange)



# Random notes
Compute radius for graphs problems as analysis


# Note on layers

Only node features:
- GCNConv: gcnconv
- SAGEConv: sageconv
- GATConv: gatconv
- GINConv: ginconv
- GeneralConv: generalconv

Edge features:
- EGATConv: egatconv
- EGINConv: eginconv


TODO with edge features:
- EGCNConv: egcnconv
- ESAGEConv: esageconv





Also edge feautes:
- SplineConv splineconv
- GeneralEdgeConv: generaledgeconv
- GeneralSampleEdgeConv: generaralsampledgeconv


This looks promising: GATv2Conv

# Datasets 
ogbg-ppa: need to test, test with normal such that dim matches on standard NN
ogbg-molhiv: techical implementation OK
ogbg-code2: NO. requires special treatement
ogbn-proteins: cannot run BFS preprocessing (graph too big)



# Random notes

agg batch does not produce correct csv

VPN trick: sudo openconnect https://sslvpn.ethz.ch/student-net --script 'vpn-slice --dump --verbose vpn-specific.host 129.132.93.64/26'

stats htop

# TODO for next experiment
- mark done
- allow to get score on best val?
- do not use dash "-" in experiment name
- registrer name with _ instead of -


MOLHIV BN norm encoder?

# Da discutere con Nat:
- Overview generale 
- Check logica codice

- Preprocessing di dataset huge per node prediciton? Ed così li carichiamo?
- Che statistiche salvare per il dataset test? Proposta: ultimo + score sul best dato dal validation
- Risultati che abbiamo ora 


Priorità:
- Test con val best
- Directed
- TUDataset
- Sintetico
