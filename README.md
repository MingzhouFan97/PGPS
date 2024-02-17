# PGPS

This is the repositry for the code of the paper **Path Guided Particle-based Sampling**. The code to reproduce the experiment results reported in the paper is uploaded in the "experiments" folder.


## Illustrative example
Here is an illustrative example showing the effectiveness of PGPS, please refer to the file example.ipynb in the root folder for the code generating this example. 

### Target

![](./independent.png)


### LD over iterations

![](./LD.gif)

<img src="LD.gif" loop==100>

<FastImage
        style={{ width: 250, height: 100, marginBottom: 50 }}
        source={require('@LD.gif')}
        resizeMode={FastImage.resizeMode.contain}
      />



### PGPS over iterations

![](./PGPS.gif)

### PGPS evolved time t change with steps

![](./pgps_time.png)