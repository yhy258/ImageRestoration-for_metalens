# NAFNet (Nonlinear Activation Free Network)
제대로 훈련시킬때는 CosineAnnealing Tmax 파라미터, Epoch 수 조정해야함 (논문 참고.)
## Dataset
1. GOPRO-Large (Deblurring)




## Experiment
1. **GOPRO-Large**  
    1. **Origin + Center Crop, Evaluation : 동일 Center Crop (Input data와 동일 Resolution)**  
        PNSR : 30.666744232177734 SSIM : 0.9074470400810242 (20epoch)
    2. **Origin + Random Crop, Evaluation : 동일 Center Crop (Input data와 동일 Resolution)**  
        PNSR : 26.735857009887695 SSIM : 0.7975642681121826 (20epoch)  
        PNSR : 26.319204330444336 SSIM : 0.7852581143379211 (10epoch)
    3. 