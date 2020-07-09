

### Building the Container

login to container registry and build the docker file

```  
docker build . -f 6d.Dockerfile -t computervisi6b3936b2.azurecr.io/6d:1
```


### References

* https://github.com/j96w/6-PACK
* https://github.com/hughw19/NOCS_CVPR2019


```BibTeX
 @InProceedings{Wang_2019_CVPR,
               author = {Wang, He and Sridhar, Srinath and Huang, Jingwei and Valentin, Julien and Song, Shuran and Guibas, Leonidas J.},
               title = {Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation},
               booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
               month = {June},
               year = {2019}

@article{wang20196-pack,
  title={6-PACK: Category-level 6D Pose Tracker with Anchor-Based Keypoints},
  author={Wang, Chen and Mart{\'\i}n-Mart{\'\i}n, Roberto and Xu, Danfei and Lv, Jun and Lu, Cewu and Fei-Fei, Li and Savarese, Silvio and Zhu, Yuke},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2020}
}


```