# Cricket Classification with Singularity

Follow these steps to set up and run the cricket classification project using Singularity:

1. **(Optional)** Create your own `.sif` image following the instructions at the [HPRC TAMU Wiki](https://hprc.tamu.edu/wiki/SW:Singularity). Alternatively, you can reuse the `dl.sif` provided. I Used this [Docker Image](https://github.com/pvbhanuteja/gs-dl-baseimage-cloudflared)

2. Change the working directory:
        ```
        cd /gpfs/proj1/choe_lab/bhanu
        ```
3. Run the Singularity command:
   ```
   singularity run --nv --fakeroot --bind ./docker_mount:/mnt dl.sif bash
   ```
4. Change to the cricket classification directory:
   ```cd /mnt/cricket-classification```

5. Edit `config.json` with your desired configuration.

6. Run the pipeline script to preprocess and train the model: ```sh run_pipeline.sh```


