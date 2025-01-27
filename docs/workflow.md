# Full Experiment Analysis Workflow

## Initialize datastore

```mermaid
%%{init: { "theme": "default", "themeVariables": { "htmlLabels": true, "curve": "linear", "layout": "elk" } } }%%
flowchart TD
 subgraph s1["Define datastore"]
        n1["Number of tiles"]
        n2["Microscope metadata"]
        n3["Experiment metadata"]
        n4["qi2labdatastore"]
  end
    n1 --> n4
    n2 --> n4
    n3 --> n4

    n1@{ shape: notch-rect}
    n2@{ shape: notch-rect}
    n3@{ shape: notch-rect}
    n4@{ shape: lin-cyl}
```

## Populate datastore

```mermaid
%%{init: { "theme": "default", "themeVariables": { "htmlLabels": true, "curve": "linear", "layout": "elk" } } }%%
flowchart TD
 subgraph s2["Place raw data into datastore"]
        n5["Raw data"]
        n12["Camera correction"]
        n6["Geometric transformation
        (local to global)"]
        n7["Experiment order"]
        n8["Fidicual data"]
        n9["MERFISH data"]
        n10["Global tile positions"]
        n11["qi2labdatastore"]
 end
    n5 --> n12
    n12 --> n6
    n6 --> n7
    n7 --> n8
    n7 --> n9
    n8 --> n11
    n9 --> n11
    n10 --> n11

    n5@{ shape: procs}
    n6@{ shape: notch-rect}
    n7@{ shape: notch-rect}
    n8@{ shape: procs}
    n9@{ shape: procs}
    n10@{ shape: procs}
    n11@{ shape: lin-cyl}
    n12@{ shape: notch-rect}
```
##  Deconvolve, register, and spot predict all tiles

```mermaid
%%{init: { "theme": "default", "themeVariables": { "htmlLabels": true, "curve": "linear", "layout": "elk" } } }%%
flowchart TD

 subgraph s5["MERFISH preprocessing"]
    direction TB
    n17["Deconvolution"]
    n18["U-FISH prediction"]
    n19["Tile warping"]
 end
 subgraph s6["Fiducial preprocessing"]
    direction TB
    n14["Deconvolution"]
    n15["Rigid registration"]
    n16["Deformable registration"]
 end
 subgraph s4["Tile N"]
    s5
    s6
 end
 subgraph s3["Register and preprocess"]
        n13["qi2labdatastore"]
        s4
 end

    s6 --> s5
    s4 <--> n13 
    n14 --> n15
    n15 --> n16
    n17 --> n18
    n18 --> n19
    n13@{ shape: lin-cyl}
    n14@{ shape: procs}
    n15@{ shape: procs}
    n16@{ shape: procs}
    n17@{ shape: procs}
    n18@{ shape: procs}
    n19@{ shape: procs}
```

## Global registration and fusion of first fiducial round

## Cell segmentation of fused first fiducial round

## Pixel decoding

## 3D segmentation based on decoded RNA