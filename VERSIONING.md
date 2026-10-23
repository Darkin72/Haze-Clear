# Quản lý file trong Git

Repo lưu mã nguồn, notebook, cấu hình mẫu và tài liệu của tất cả các phase. Notebook được lưu không kèm output. Các file sinh từ một lượt chạy vẫn ở trên máy và không được đưa vào commit:

- Trọng số, checkpoint, model export và file nén tải về.
- Cache Python, cache notebook, môi trường ảo và log.
- Thư mục `datasets/`, nội dung tự sinh trong `results/` ở cấp phase, ảnh dự đoán và ảnh lỗi.
- CSV đo đạc, JSON mô tả runtime và `config.json` ở cấp phase; đây là cấu hình của lượt chạy cụ thể, có thể chứa đường dẫn Colab.

Các quy tắc này áp dụng cho phase 1, 2, 2.1, 3, 3.1, 3.2 và 4. Cấu hình huấn luyện trong `options/` vẫn được theo dõi.

Bản lưu báo cáo có chủ đích được giữ tại [phase 4/reported_20261021](phase%204/reported_20261021/README.md), cùng [bảng đối chiếu hai lượt](phase%204/results/bao_cao_da_chot.md) và [hướng dẫn đọc số liệu](phase%204/REPORTING.md). Bản lưu có manifest và nguồn log để phân biệt số liệu lượt cũ với lượt chạy lại. Các đường dẫn tới artifact lượt mới chỉ truy cập được khi có output tương ứng trên máy.

`.gitignore` không xóa file trên máy. Với cache đã từng được commit, dùng `git rm --cached` để bỏ theo dõi mà vẫn giữ bản local. Repo này có remote riêng; commit và push tại chính thư mục `solution/Haze Clear`, không thêm nó như một repo lồng vào repo cha.
