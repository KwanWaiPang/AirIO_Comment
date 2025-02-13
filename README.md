[comment]: <> 

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability
  </h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://kwanwaipang.github.io/AirIO/">Blog</a> 
  | <a href="https://github.com/Air-IO/Air-IO">Original Github Page</a>
  | <a href="https://github.com/KwanWaiPang/AirIMU_comment">AirIMU_comment</a>
  </h3>
  <div align="justify">
  </div>

<br>

<!-- ~~~
rm -rf .git
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/KwanWaiPang/AirIO_Comment.git
git push -u origin main
~~~ -->

# 配置安装
```bash
conda create -n airio python=3.10.11
conda activate airio

#同样依赖于pypose
# 安装系列依赖
pip install -r requirements.txt

```

# 下载数据集
* [Euroc](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets),路径在```/home/gwp/AirIMU/Euroc_dataset```
* [Blackbird dataset](http://blackbird-dataset.mit.edu/BlackbirdDatasetData/),路径在```/home/gwp/Air-IO/Blackbird_dataset```