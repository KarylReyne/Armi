THF Airport ArgMining Corpus (version 1.0, 2016-09-08)
**********************************************************************
This archive contains the raw corpus from the online participation project 'Tempelhofer Feld' and the annotations that were created for the publication titled 'What to Do with an Airport? Mining Arguments in the German Online Participation Project Tempelhofer Feld'.


== Background ==
The online participation project is available at https://tempelhofer-feld.berlin.de/i/tempelhofer-feld/category 
The dataset comprises proposals and comments which each have a unique identifier.
Example proposal: https://tempelhofer-feld.berlin.de/i/tempelhofer-feld/proposal/66-Trinkwasserbrunnen_auf_dem_Tempelhofer_F
More background information can be found in our publication http://aclweb.org/anthology/W/W16/W16-2817.pdf


== Content ==
This archive contains the following files:
(1) Raw corpus as JSON file from the Adhocracy 2 instance at https://tempelhofer-feld.berlin.de/i/tempelhofer-feld/category (2015-07-07 dump)
(2) Our annotations in the brat rapid annotation tool (You don't need to extract the data from the annotation files yourself. Use (3) instead.)
(3) Our training/test split for both subtasks, including raw sentences
    Subtask A: Classify sentences as argumentative or non-argumentative
    * Size: training 1947 sentences, test: 486 sentences
    * Labels: argumentative/non-argumentative
    Subtask B: Classify argument components in argumentative sentences with exactly one annotated argument component
    * Size: training: 1592 sentences, test: 398 sentences
    * Labels: MajorPosition/ClaimPro/ClaimContra/Premise (We grouped 'ClaimPro' and 'ClaimContra' into 'Claim' for our evaluation. More details can be found in the paper)


== Citation ==
If you use the dataset, please cite the following paper:

@InProceedings{liebeck-esau-conrad:2016:ArgMining2016,
  author    = {Liebeck, Matthias  and  Esau, Katharina  and  Conrad, Stefan},
  title     = {{What to Do with an Airport? Mining Arguments in the German Online Participation Project Tempelhofer Feld}},
  booktitle = {Proceedings of the Third Workshop on Argument Mining (ArgMining2016)},
  month     = {August},
  year      = {2016},
  publisher = {Association for Computational Linguistics},
  pages     = {144--153},
  url       = {http://www.aclweb.org/anthology/W16-2817}
}


== License ==
Creative Commons License (CC BY-SA 3.0)

== Contact Person ==
Matthias Liebeck, liebeck@cs.uni-duesseldorf.de, https://dbs.cs.uni-duesseldorf.de/mitarbeiter.php?id=liebeck