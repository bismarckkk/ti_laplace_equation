## taichi实现拉普拉斯方程基本解模拟
对于二维平面理想不可压流体的定常无旋流动中，其流函数 $\Psi (x, y)$ 满足以下方程  
$$\frac{\partial^2 \Psi}{\partial x^2}+\frac{\partial^2 \Psi}{\partial y^2}=0$$  
即拉普拉斯方程，这个demo可以模拟点源、点汇、点涡、偶极子等拉普拉斯方程的基本解，
即理想不可压流体的一些最基本的平面定常无旋流动  
在这个demo中，可以交互式的创建或更改上述基本解，通过箭头和从边缘出发的物质点显示该流场中的流动情况

### 运行
```shell script
python demo.py
```
* 按下`s`/`v`/`d`键后在gui中按下鼠标左键或右键，将在鼠标位置创建一个点源/点涡/偶极子
或强度为负数的点源（点汇）/点涡/偶极子
* 在已经创建的点源/点涡/偶极子附近同时按下鼠标左键和右键将删除该对象
* 直接在点源/点涡/偶极子附近按下鼠标左键将增加其强度的绝对值；若按下右键将减少其强度的绝对值

### 示例截图
![demp.png](demo.gif)  
库塔-茹科夫斯基升力定理流动示例
