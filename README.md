# ttcsnhandienboicanh
yolo ko nhận file json nên cần chuyển đổi dữ liệu coco sang .txt chạy file convert_coco_to_yolo.py
sau khi chuyển đổi sang sẽ có file datasets có nhãn và ảnh trong đó của val và train
 e sẽ lấy 1/20 só ảnh đẻ train và val vì máy em chạy hơi lâu  generate_datasetsmaill.py
 rồi chia 80 để train 20 để val từ file trên sinh ra trong datasmaill
 cấu tạo file coco128 chỉ có 80 nhãn nên sẽ có nhiều ảnh  ko có nhãn trong số đó ta cần loại  bỏ chúng em chạy file test.py để loại bố 
 rồi em chạy train_model.py kia để train yolov8 sau khi train xog sẽ  hiện ra model  best và last trong file detect và sinh ra file yolo11n.pt để  dự đoán
 cuối cùng và file app là chạy thử nghiệm 
 gg drive datasetsmaill : https://drive.google.com/drive/folders/11yp_NzALseMznpMhmr1zESedHrp4N4jH
2 datasetcoco và dataset kia  dung lượng lớn quá ko up đc : ![image](https://github.com/user-attachments/assets/14469ea1-4144-496a-909b-2987be19a95a)
