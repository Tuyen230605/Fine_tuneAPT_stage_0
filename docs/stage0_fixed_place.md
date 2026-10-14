# OpenArm: fine tune APT Stage 0 với `dat_vat`

Quy trình này chuyển episode raw sang HDF5 mà APT đọc được, tự nhận diện tay
đang hoạt động, rồi fine tune từ checkpoint VA Stage 0. Raw data và lệnh chạy
đều ở máy train; repo mang theo converter, adapter và URDF.

## Episode mẫu và tay trái/phải

Episode mẫu có 675 frame ở 50 Hz, 16 joint values, camera `chest` và `front`
(RGB 480×640). `action` giống hệt `observations/qpos`; `joint_commands` có
NaN, vì vậy converter dùng qpos làm trạng thái thật tương lai.

Bạn xác nhận có 50 episode mỗi tay, mỗi episode chỉ thao tác một tay và raw
không ghi nhãn tay. Converter tính FK bằng URDF rồi so quãng đường TCP hai
tay. Chỉ nhận diện khi một tay đi ít nhất 5 cm, dịch chuyển khỏi điểm đầu ít
nhất 3 cm, và quãng đường lớn ít nhất 3 lần tay kia. Episode không rõ sẽ dừng
với hai số đo để tránh gán nhãn bừa. Có thể sửa bằng file override JSON như
`{"episode_001.hdf5":"left"}`; key là đường dẫn tương đối từ input root.

Episode mẫu được nhận diện là **tay phải**: TCP trái đi 1.5 cm, TCP phải đi
1.65 m. Converter ghi cả hai EE theo đúng thứ tự trái/phải, nhưng chỉ bật
`valid_ee_mask` cho tay đang làm. APT chỉ tính loss trên tay đó.

## Ma trận camera: mặc định không cần điền

APT dùng `K` để đổi pixel thành tia nhìn và `base_T_camera` để biết pose
camera trong hệ base robot. Raw không có thông tin hiệu chuẩn, nên template
dùng chế độ `approximate_static` để bạn chạy data/train ngay:

- `K` ước lượng tiêu cự 640 px và tâm ảnh ở giữa khung hình.
- Pose của cả hai camera tạm đặt identity trong hệ base.
- Episode output ghi rõ `camera_geometry_mode=approximate_static` và giá trị
  tiêu cự giả định.

Đây là xấp xỉ để thử fine tune với camera cố định; nó không phải hiệu chuẩn đo
được và có thể làm prior hình học của APT kém chính xác. Khi có calibration,
đổi `camera_geometry_mode` sang `calibrated`, điền `K`, hệ số méo OpenCV `D`
(hoặc `[]` nếu ảnh đã rectified), và `base_T_camera` thực cho từng camera rồi
chuyển đổi lại với `--overwrite`. `base_T_camera` phải là pose optical frame
trong `openarm_body_link0`, đơn vị mét.

### Cách đo để thay giá trị xấp xỉ

Nếu robot chạy ROS 2, trước hết thử lấy dữ liệu đã có:

```bash
ros2 topic echo /camera/color/camera_info
ros2 run tf2_ros tf2_echo openarm_body_link0 camera_optical_frame
```

Thay đúng tên topic/frame. Lệnh đầu có `K,D`; lệnh sau in transform cần cho
`base_T_camera`. Chỉ dùng `camera_info` nếu topic đó có cùng độ phân giải,
focus và ảnh raw được ghi vào HDF5. Với ảnh đã rectify, lấy fx/fy/cx/cy từ
projection matrix `P` và đặt `D: []`.

1. **Đo `K,D` riêng cho từng camera.** In checkerboard có kích thước ô đã biết
   (chẳng hạn 25 mm), chụp khoảng 20–30 ảnh ở nhiều vị trí/góc, phủ đều khung
   hình. Giữ đúng độ phân giải 640×480 và chế độ focus sẽ dùng lúc chạy.
   OpenCV `calibrateCamera` trả về `K` và `D`. `K` có dạng
   `[[fx,0,cx],[0,fy,cy],[0,0,1]]`, các giá trị tính bằng pixel.
2. **Đo extrinsic của từng camera.** Nếu ROS TF đã có transform từ
   `openarm_body_link0` tới `*_camera_optical_frame`, dùng trực tiếp transform
   đó. Nếu chưa có, dùng một bảng AprilTag/ChArUco gắn cứng lên TCP hoặc một
   jig có vị trí đo được trong base. Với TCP, lấy `base_T_tcp` từ qpos qua
   URDF, đo transform lắp bảng `tcp_T_board`, rồi tính
   `base_T_board = base_T_tcp × tcp_T_board`.
3. Chụp bảng ở ít nhất 10 pose robot khác nhau. OpenCV `solvePnP` trên mỗi
   ảnh cho `camera_T_board` (pose board trong camera). Tính
   `base_T_camera = base_T_board × inverse(camera_T_board)` cho từng pose,
   sau đó lấy trung bình pose và kiểm tra độ lệch. Dùng bảng có marker xác
   định hướng để tránh nhầm góc 180 độ.
4. Vẽ lại các điểm board/TCP lên ảnh bằng `K,D` và `base_T_camera`. Nếu điểm
   không trùng vật thật trên ảnh, kiểm tra hướng transform, optical frame,
   đơn vị mét và đồng bộ timestamp trước khi đổi sang chế độ `calibrated`.

## Chuẩn bị Ubuntu trên máy train

Ở thư mục repo:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Ví dụ PyTorch/CUDA 12.6 như README; chọn wheel phù hợp driver máy train.
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install 'h5py>=3.16'
```

Lệnh train dùng DDP. Stage 0 vẫn tải Qwen3-VL 2B làm encoder ảnh dù language
bị mask, nên máy train cần truy cập Hugging Face lần đầu và có đủ VRAM.

Checkpoint đang ở máy code tại
`checkpoints/APT/apt_va/ckpt_latest.pt`; thư mục này bị Git ignore. Chép file
checkpoint sang cùng đường dẫn trên máy train. Nếu tải lại theo README thì
kiểm tra bằng:

```bash
python scripts/test_ckpt.py --ckpt checkpoints/APT/apt_va/ckpt_latest.pt --train-stage 0
```

## Chuyển dữ liệu

Đặt toàn bộ 100 episode raw dưới một thư mục, ví dụ `/data/openarm_raw`. Dùng
URDF và config mẫu đã có sẵn trong repo, không cần sửa config ở chế độ xấp xỉ:

```bash
python scripts/convert_openarm_episode.py \
  --input-root /data/openarm_raw \
  --output-root /data/openarm_apt \
  --config configs/openarm_fixed_place.template.json \
  --val-fraction 0.1
```

Converter tự tạo `train/` và `val/`, giảm 50 Hz xuống khoảng 10 Hz, tính pose
TCP, chuẩn hóa gripper 0–1 và nén ảnh JPEG. Cuối lệnh có thống kê số episode
trái/phải. Nếu gặp episode mơ hồ, thêm mapping vào `/data/arm_overrides.json`
rồi chạy lại với `--arm-overrides /data/arm_overrides.json --overwrite`.

Kiểm tra output và một mẫu qua chính adapter APT:

```bash
python scripts/check_openarm_dataset.py /data/openarm_apt --apt-sample
```

Kiểm tra episode count trái/phải, `active` và `valid_ee_mask`. Bộ kiểm tra APT
xác nhận prompt `dat_vat`, ảnh 2 camera, hai slot EE, mask một tay và chuỗi 32
target tương lai.

## Train Stage 0

```bash
export APT_FIXED_PLACE_DATA_GLOB='/data/openarm_apt/train/**/*.h5'
bash scripts/finetune.sh --backend ddp --gpus 0 --stage 0 \
  --config finetune_openarm_fixed_place \
  --pretrained-ckpt checkpoints/APT/apt_va/ckpt_latest.pt \
  --va-name ft_va_dat_vat --bs 1 --max-iter 25000 --save-interval 5000
```

Đổi `--gpus` theo máy train. Nếu hết VRAM, giữ batch size 1 và bật gradient
checkpointing qua cấu hình; nếu chạy ổn có thể tăng batch size. Lệnh trên tạo
run mới từ weights Stage 0. Để resume, dùng `--va-conti ft_va_dat_vat` thay
cho `--pretrained-ckpt` và `--va-name`. Checkpoint nằm ở
`checkpoints/APT/ft_va_dat_vat/`.

Mọi episode có prompt cố định `dat_vat`. Attention mask Stage 0 chặn language
khỏi nhánh vision/action; hành động không được điều kiện hóa bởi prompt. Chỉ
train trên `train/`; `val/` để đánh giá episode chưa thấy và đo tỷ lệ gắp/đặt
thành công.
