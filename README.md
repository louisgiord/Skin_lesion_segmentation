# 4IM01-SkinLesions-GiordmainaBonniniere

***

## Name
Skin Lesion Detection and Segmentation of Medical Images.

## Description
The main aim of the project is to build a pipeline taking a skin lesion image and returning the most efficient mask of the lesion using segmentation methods. It also includes parts of pre and post processing in order to generate the most efficient mask of the lesion. 

## Roadmap
26/09/24 : First meeting with our supervisor M. Pietro Gori. He explained the project to us, the goals and presented the different resources available. 
For next meeting : reading the papers and coming up with a roadmap to organize as well as possible the work, dividing task. 

Papers reading : Louis read the pre-processing paper "Shading Attenuation in Human Skin Color Images""Border DEtection in dermoscopy images using statistical region merging", and "Dermoscopic skin lesion image segmentation based on Local Binary Pattern Clustering : Comparative Study". Furthermore, Louis had an overlook view of the other papers. Maëliss read "Dermoscopic skin lesion image segmentation based on Local Binary Pattern Clustering : Comparative Study" and Dull razor. 

03/10/24 : Choice of the implementation strategy. Questions on the articles. 
Le choix s'est porté sur une première implémentation de la méthode d'Otsu (Louis) et du Dull Razor comme hair remover pour Maëliss.
Implémentation du Dull Razor décalée après la séance du 21/10, car le cours sur les noyaux morphologiques n'a pas encore été vu.
Etude du choix de l'espace couleur.

10/10/24 : Otsu method is functionning. Next goal is to implement fully the pre-processing, especially the hair removal and the black frame removal. 
For morphologic maths, we can start to read the course that has not yet be done. For black frame removal, remove all the black columns and lines of the image (meaning erasing them from the original image), then using method of region merging starting with 4 sets of pixels in the four corners of the image (i.e if their is a black lense in the image)
Furthermore, compute the DICE score to start comparing our results with the ground-truth masks, by having a qualitative and quantitative analysis of the pipeline. Use the ablation study method in order to understand how the different blocks( preprocessing, segmentation algorithm, and post-processing) improve or not the DICE score. 

17/09/24 : goal = finalizing the pre-processing in order to then start the segmentation part itself. 
Work divided between the both of us : 
- Black removal (and if enough time finishing shading attenuation) for Louis
- Hair removal for Maëliss

