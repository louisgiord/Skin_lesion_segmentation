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

17/09/24 : goal = finalizing the pre-processing in order to then start the segmentation part itself. 
Work divided between the both of us : 
- Shading attenuation for Louis
- Hair removal for Maëliss
- Black removal for both of us

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.telecom-paris.fr/Student_Proects/4im01-skinlesions-giordmainabonniniere.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.telecom-paris.fr/Student_Proects/4im01-skinlesions-giordmainabonniniere/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)
