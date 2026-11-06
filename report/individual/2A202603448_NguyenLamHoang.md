# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** `7114`.
- **Thử thách cá nhân:** đề #5 (`SEED % 6 = 4`, đánh số đề từ 1).
- Thông tin định danh được thể hiện trong tên file theo quy ước nộp bài; nội dung tuân theo yêu cầu bảo mật của `RULES.md`.

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) | Đã có notebook, code, câu trả lời và output thí nghiệm. |
| [`lab1_report.json`](lab1_report.json) | Đã có báo cáo xuất từ notebook: SEED, 10 dự đoán, kết quả kiểm tra và bảng so sánh. |
| [`hand_calc.jpg`](hand_calc.jpg) | Đã có ảnh viết tay với các bước của 5 phép tính, gửi kèm để giảng viên chấm. |
| [Side lab: solution.py](../../src/optional_llm_api/solution/solution.py) và [exercises.md](../../src/optional_llm_api/solution/exercises.md) | Đã triển khai 7 hàm, trả lời phản ánh; bộ test mock của lớp đạt 19/19.|

Theo JSON đã cập nhật, **15/15 mục kiểm tra đều đạt**, gồm **10 mục code và 5 mục tính tay**. Em đã hoàn thành phần tính tay và đính kèm `hand_calc.jpg` với các bước giải để giảng viên đối chiếu, chấm bài.

Dữ liệu ban đầu có 5.572 tin, trong đó 403 tin trùng tuyệt đối. Sau khi loại trùng còn **5.169 tin: 653 spam và 4.516 ham**. Với SEED trên, tập train có 3.619 tin, validation có 775 tin, test có 775 tin (104 spam); SUB gồm 40 tin lấy từ test, cân bằng 20 spam và 20 ham. Nếu luôn đoán ham thì accuracy trên toàn bộ dữ liệu đã bỏ trùng vẫn đạt khoảng **87,37%**, nhưng recall và F1 của lớp spam bằng 0; vì vậy accuracy riêng lẻ chưa đủ để đánh giá bộ lọc.

## 2. Kết quả chính

Các số dưới đây lấy từ `lab1_report.json`, làm tròn để trình bày. Lớp dương là spam. F1 test của các mô hình ML/DL/Embedding được đo trên 775 tin; LLM chỉ chạy trên SUB nên không có kết quả trên toàn bộ test.

| Phương án | F1 test (775 tin) | F1 SUB (40 tin) | Train (s) | Thông lượng cả lô (tin/s) | p95 (ms/tin) | USD/1.000 tin trong lần chạy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0,8995 | 0,8571 | 6,54 | 23.560,35 | 0,153 | 0 |
| 2. BoW + MLP | 0,9400 | 0,9474 | 7,95 | 25.285,80 | 2,383 | 0 |
| 3a. MiniLM + kNN | 0,8900 | 0,9474 | 75,30 | 47,15 | 56,451 | 0 |
| 3b. MiniLM + LogReg | 0,9119 | 0,9189 | 75,74 | 49,13 | 48,755 | 0 |
| 3c. MiniLM + MLP head | 0,9146 | 0,9189 | 76,54 | 53,51 | 33,083 | 0 |
| 4a. LLM nhỏ zero-shot | — | 0,9231 | 0 | 1,12 | 2.411,390 | 0 |
| 4b. LLM lớn zero-shot | — | 0,9231 | 0 | 1,22 | 2.415,635 | 0 |
| 4c. LLM nhỏ few-shot | — | 0,9000 | 0 | 0,63 | 3.028,185 | 0 |

Phần LLM thực tế dùng **Groq**, với model nhỏ `openai/gpt-oss-20b` và model lớn `openai/gpt-oss-120b`. Em dùng gói API miễn phí nên chi phí thực trả của lần thực hành là **0 USD**. Trong code, hai model Groq cũng chưa có trong `PRICE_PER_1M`, nên hàm ước tính mặc định trả 0; bảng này chưa cung cấp đơn giá để ngoại suy chi phí khi triển khai vượt phạm vi gói miễn phí. Với các mô hình chạy cục bộ, số 0 chỉ nói về phí API, chưa tính tài nguyên máy.

BoW + MLP đạt F1 test cao nhất trong các phương án được đo trên toàn bộ test: **0,9400**, precision **0,9792**, recall **0,9038**. So với BoW + LogReg có precision **1,0000** nhưng recall **0,8173**, MLP bắt được nhiều spam hơn và chấp nhận thêm một ít báo nhầm. Trên cùng embedding, thêm MLP head chỉ tăng F1 từ **0,9119 lên 0,9146**, nhỏ hơn mức tăng từ **0,8995 lên 0,9400** trên BoW; điều này phù hợp với việc encoder đã cung cấp biểu diễn giàu thông tin trước khi phân loại.

Để so với LLM, em dùng cột **F1 SUB**. SUB chỉ có 40 tin nên chênh lệch vài điểm F1 chưa đủ để khẳng định ưu thế ổn định. Biểu đồ trong notebook dùng `1 − F1 SUB` và p95 cho thấy **BoW + LogReg và BoW + MLP nằm trên đường Pareto**; `1 − F1` ở đây không phải tỷ lệ dự đoán sai.

## 3. Dự đoán và thực tế

Em chọn ba dự đoán có kết quả lệch đáng chú ý. Thời điểm ghi dự đoán được giữ nguyên trong JSON; em không sửa dự đoán sau khi biết kết quả.

| Dự đoán | Em dự đoán trước thí nghiệm | Kết quả và giải thích |
| --- | --- | --- |
| `ml_lr` — 15:15:53 | `lr=50` có thể khiến loss dao động mạnh hoặc tăng do bước cập nhật quá lớn. | Theo phần nhận xét thí nghiệm trong notebook, loss tăng mạnh ban đầu lên khoảng 2,5 nhưng cuối cùng giảm còn khoảng **0,006** sau 300 epoch; `lr=0,001` còn khoảng 0,626 và `lr=0,5` còn khoảng 0,083. Bước lớn gây mất ổn định ban đầu nhưng trong lần chạy này vẫn hội tụ. Em hiểu rằng learning rate lớn là một nguy cơ, không đồng nghĩa chắc chắn phân kỳ; train loss thấp cũng chưa chứng minh tổng quát hóa tốt. |
| `llm_models` — 15:21:41 | Model lớn sẽ có F1 cao hơn model nhỏ khoảng 0,02–0,05. | Hai model zero-shot đều đạt **F1 SUB = 0,9231**, precision 0,9474 và recall 0,9000. Trong tập nhỏ này, tăng kích thước model chưa tạo ra cải thiện đo được; cùng chỉ số tổng hợp cũng chưa chứng minh hai model sai đúng những tin giống nhau. |
| `llm_fewshot` — 15:22:54 | Bốn ví dụ sẽ giúp F1 tăng nhẹ. | F1 giảm từ **0,9231 xuống 0,9000**, còn p95 tăng từ **2.411,390 lên 3.028,185 ms/tin**. Các ví dụ có thể làm thay đổi ranh giới phân loại nhưng chưa chắc phù hợp mọi tin; cần phân tích thêm lỗi và thử trên tập lớn hơn. Phần dự đoán chi phí tăng chưa thể kiểm chứng bằng số tiền thực trả vì lần chạy dùng API miễn phí. |

Một kết quả bổ sung là ở thử nghiệm bỏ ReLU, F1 vẫn đạt **0,940**. Hai lớp Linear liên tiếp có thể gộp thành một phép biến đổi tuyến tính, nhưng việc đó không buộc kết quả huấn luyện phải giống hệt LogReg ở Phần 1 vì cách tối ưu và quá trình huấn luyện khác nhau.

## 4. Phân tích 50 case khó

Bảng sau lấy từ output stress test của notebook, trên **50 case gốc**. Sáu case trong `MY_CASES` được thêm ở ô sau; chưa có output đánh giá lại 56 case trong bản notebook hiện tại, nên em chưa đưa chúng vào kết quả này.

| Phương án | F1 SUB thường | F1 trên 50 case khó | Accuracy trên 50 case khó |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0,857 | 0,378 | 54% |
| BoW + MLP | 0,947 | 0,500 | 60% |
| MiniLM + kNN | 0,947 | 0,772 | 74% |
| MiniLM + LogReg | 0,919 | 0,640 | 64% |
| MiniLM + MLP head | 0,919 | 0,625 | 64% |
| LLM nhỏ zero-shot | 0,923 | 0,903 | 88% |
| LLM nhỏ + prompt phòng thủ | — | 0,918 | 90% |

Prompt phòng thủ chỉ được đo trên bộ khó; giá trị SUB 0,923 mà bảng notebook đặt cạnh hàng này được lấy lại từ zero-shot, không phải phép đo riêng cho prompt phòng thủ.

**BoW + LogReg tụt nhiều nhất**, từ F1 SUB khoảng 0,857 xuống 0,378, giảm khoảng **0,479**. Theo bảng nhóm, model chỉ đúng **1/8 case thao túng/lừa đảo**, **1/6 case né từ khóa** và **2/10 case spam giọng tự nhiên**. BoW phụ thuộc từ trong vocab, nên dễ bỏ sót câu lừa đảo viết giống trò chuyện hoặc thay ký tự. Đây là so sánh giữa hai bộ dữ liệu khác nhau về nội dung và tỷ lệ lớp; mức giảm giúp phát hiện điểm yếu, chưa phải ước lượng độ giảm ngoài thực tế.

Embedding xử lý được một số biến thể ngữ nghĩa nhưng vẫn gặp khó khi phân biệt spam với câu đang kể lại spam: kNN chỉ đúng **25%** nhóm `talks_about_spam`, còn LogReg và MLP head đúng **50%**. LLM nhỏ zero-shot yếu nhất ở nhóm `ham_looks_spam`, đúng **60%**. Hai lỗi cụ thể được output xác nhận ở LLM nhỏ dùng prompt phòng thủ:

1. **Case #2 — ham bị đoán thành spam:** “FREE pizza in the office kitchen right now, come quick before it's gone”. Đây là lời mời ăn pizza trong văn phòng. Một cách giải thích hợp lý là từ “FREE” cùng lời giục “come quick” làm model đánh giá quá mức dấu hiệu quảng cáo, dù ngữ cảnh nội bộ là vô hại.
2. **Case #47 — ham bị đoán thành spam:** tin kể lại việc nhận thông báo trúng tiền và chế giễu trò lừa đảo. Model có thể chú ý vào nội dung trúng thưởng, gọi điện được trích lại, nhưng bỏ qua ý người gửi đang nói **về** spam. Đây là lỗi phân biệt nội dung được nhắc đến với ý định của cả tin nhắn.

Prompt phòng thủ tách SMS trong thẻ `<sms>` và yêu cầu coi nội dung bên trong là dữ liệu. Accuracy nhóm prompt injection tăng từ **5/6 lên 6/6**, tổng accuracy tăng từ **88% lên 90%**; đồng thời nhóm spam giọng tự nhiên giảm từ **100% xuống 90%**. Như vậy bản sửa có ích trên bộ này nhưng vẫn có đánh đổi. Có thể bổ sung bộ phân loại độc lập và kiểm tra trường hợp bất đồng; với BoW, chuẩn hóa ký tự thay thế hoặc chữ bị ngăn cách có thể giúp, nhưng phải tránh làm hỏng mã và nội dung hợp lệ.

Bộ 50 tin được thiết kế để tìm lỗi, không đại diện đầy đủ cho lưu lượng SMS thật. Em dùng nó để phân tích khả năng tổng quát hóa và kiểm tra bản sửa, tránh kết luận hiệu quả triển khai chỉ từ bộ này.

## 5. Thử thách cá nhân

**Đề #5:** viết 4 tin nhắn gồm 2 spam né từ khóa và 2 ham chứa `free`/`win`; so sánh BoW với Embedding + LogReg rồi giải thích từng tin.

Em dùng cùng vocab và trọng số đã học ở Phần 1 để dự đoán BoW. Với embedding, em encode bằng MiniLM rồi chuẩn hóa bằng `mu`, `sd` đã tính từ train trước khi đưa vào LogReg; cả hai dùng ngưỡng 0,5.

| Tin nhắn trong ô thử thách | Nhãn thật | BoW + LogReg | MiniLM + LogReg | Nhận xét |
| --- | --- | --- | --- | --- |
| Your parcel is on hold. Confirm your card details at parcel-update.example to arrange delivery. | spam | ham | ham | Cả hai bỏ sót yêu cầu cung cấp thông tin thẻ được ngụy trang thành thông báo giao hàng. |
| Hi, I found a private group that doubles your deposit in three days. Send me your number and I'll invite you. | spam | ham | ham | Cả hai bỏ sót lời dụ đầu tư với hứa hẹn nhân đôi tiền, được viết theo giọng người quen. |
| Are you free after class? Let's get lunch together. | ham | ham | ham | Cả hai đúng; `free` đang nói về thời gian rảnh trong lời rủ đi ăn. |
| We can win tomorrow's football match if we practise today. | ham | ham | ham | Cả hai đúng; `win` nói về trận bóng, không phải mồi trúng thưởng. |

Output ghi nhận **BoW đúng 2/4 và Embedding + LogReg đúng 2/4**, cả hai đoán toàn bộ là ham. Khả năng biểu diễn ngữ nghĩa của encoder chưa đảm bảo head phân loại nhận ra kiểu spam mới. Kết quả cũng cho thấy BoW không cứ gặp `free` hoặc `win` là đoán spam: quyết định phụ thuộc tổng trọng số của các từ trong tin. Các giải thích về nguyên nhân bỏ sót ở đây là suy luận từ nội dung và cơ chế mô hình, chưa phải phân tích đóng góp từng đặc trưng.

## 6. Phương án cho sản phẩm thật

Với yêu cầu **10 triệu SMS/ngày, p95 < 50 ms/tin**, em chọn **BoW + MLP làm bộ lọc ban đầu**. Phương án này đạt F1 test **0,9400**, p95 **2,383 ms/tin**, train khoảng **7,95 giây** và thông lượng cả lô khoảng **25.286 tin/giây** trên môi trường đo. Lưu lượng trung bình cần xử lý là `10.000.000 / 86.400 ≈ 115,74 tin/giây`; số đo cả lô cho thấy tiềm năng đáp ứng, nhưng vẫn cần kiểm thử tải dịch vụ thật, tải đỉnh và hàng đợi.

BoW + LogReg, BoW + MLP, MiniLM + LogReg và MiniLM + MLP head đều đạt ngưỡng p95 trong phép đo. MiniLM + kNN vượt ngưỡng ở **56,451 ms**; cả ba phương án LLM đều trên **2.400 ms**. Hai phương án embedding có head đạt độ trễ nhưng thông lượng cả lô đo được chỉ khoảng **49–54 tin/giây**, nên cần tăng tài nguyên hoặc tối ưu nếu xử lý toàn bộ lưu lượng bằng chúng.

Đánh đổi lớn nhất của lựa chọn BoW + MLP là **stress test chỉ đạt F1 0,500 và accuracy 60%**. Trước khi triển khai, cần bổ sung dữ liệu lừa đảo mới, kiểm tra báo nhầm tin giao dịch, theo dõi thay đổi phân phối và chọn ngưỡng trên validation. Số đo tốt trên test thường chưa đủ để bỏ qua những lỗi này.

Em đề xuất dùng MLP trước, chuyển tin có xác suất trong khoảng `[0,3; 0,7]` hoặc có tín hiệu rủi ro sang bước kiểm tra bằng LLM với prompt phòng thủ. Khoảng xác suất này là thiết kế đề xuất, chưa được đánh giá trong lab; mô hình vẫn có thể sai một cách tự tin nên cần thêm giám sát và dữ liệu phản hồi. Vì LLM chậm vài giây, nhánh kiểm tra cần cơ chế xử lý riêng; không thể mặc định hệ thống lai vẫn đạt SLA 50 ms nếu chờ LLM đồng bộ.

Trong buổi thực hành, **Groq miễn phí nên chi phí thực trả bằng 0**. Với triển khai 10 triệu tin/ngày trong 30 ngày, có 300 triệu tin/tháng; nếu đơn giá thực tế là `C USD/1.000 tin`, chi phí API cho toàn bộ lưu lượng là **`300.000 × C USD/tháng`**. Nếu chỉ chuyển tỷ lệ `r` sang LLM thì phần API tương ứng khoảng `300.000 × r × C`, chưa tính chi phí hệ thống khác. Bài lab chưa xác minh hạn mức miễn phí có đáp ứng lưu lượng này, nên em không coi mức 0 của lần chạy là cam kết chi phí vận hành ở quy mô sản phẩm.

## 7. AI log và điều đã học

| Phần | Em hỏi AI gì | AI hỗ trợ gì | Em đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Lab API buổi sáng | Cách tổ chức bài làm, gọi API và xử lý lỗi. | Hỗ trợ code từ template, đọc `.env`, chạy thử và giải thích lỗi API. | Kiểm tra Gemini chạy thành công, OpenAI hết credit; yêu cầu khôi phục test gốc và bỏ test AI tự thêm. |
| Nộp bài buổi sáng | Cách xử lý lỗi quyền và đăng nhập GitHub. | Hướng dẫn đổi remote, đăng nhập bằng mã thiết bị. | Em đăng nhập và push thành công; kiểm tra `.env` không được Git theo dõi. |
| Báo cáo Lab 1 | Nhờ đọc notebook, JSON và tổng hợp kết quả. | Hỗ trợ viết báo cáo, so sánh mô hình và phân tích lỗi từ output đã có. | Em xác nhận Groq miễn phí, cập nhật JSON với 5 kiểm tra tính tay đều đạt và đối chiếu ảnh bài làm. |

Em hiểu rõ hơn sự khác nhau giữa BoW, MLP và embedding: biểu diễn tốt vẫn cần bộ phân loại và dữ liệu phù hợp. Kiến thức gọi API buổi sáng giúp em hiểu cách dùng SDK tương thích OpenAI với Groq ở buổi chiều. Kết quả thực nghiệm cho thấy model lớn hoặc few-shot chưa chắc tốt hơn; cần chọn phương án dựa trên F1, độ trễ, chi phí và khả năng xử lý case khó.
