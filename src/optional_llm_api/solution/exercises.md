# Phản ánh — Side lab LLM API

## 2.1. Độ nhạy của temperature


Với chatbot hỗ trợ khách hàng, em chọn khoảng **0–0,2** làm điểm khởi đầu để ưu tiên sự nhất quán, rồi đánh giá trên câu hỏi thật. Câu trả lời vẫn cần dựa vào tài liệu hỗ trợ đã kiểm chứng, thay vì chỉ giảm temperature để tránh sai.

Để thực hiện thí nghiệm khi có tài khoản API phù hợp, chạy từ gốc repo:

```python
from src.optional_llm_api.solution.solution import call_openai

prompt = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."
for temperature in [0.0, 0.5, 1.0, 1.5]:
    response, latency = call_openai(prompt, temperature=temperature, top_p=1.0)
    print(temperature, response, f"{latency:.3f}s")
```

Giữ `top_p=1.0` trong phép so sánh để tập trung quan sát ảnh hưởng của temperature. Code giữ model và tham số theo template bài tập; bốn lệnh trên gọi API thật, tách biệt với bộ test mock.

## 2.2. Đánh đổi chi phí

Theo **đơn giá giả định trong đề**, model lớn có giá input/output lần lượt 1,00/5,00 USD mỗi triệu token, model nhỏ là 0,10/0,50 USD. Cả hai đơn giá đều chênh **10 lần**, nên với cùng lượng input và output, model lớn đắt gấp 10 lần.

Có `10.000 × 3 = 30.000` lần gọi/ngày và `30.000 × 350 = 10.500.000` token/ngày. Đề chưa tách 350 token thành input và output; gọi `I` và `O` là số triệu token input/output mỗi ngày thì:

- Model lớn: `I × 1,00 + O × 5,00` USD/ngày.
- Model nhỏ: `I × 0,10 + O × 0,50` USD/ngày.

Nếu giả sử mỗi lần gồm 250 token input và 100 token output, thì `I = 7,5`, `O = 3`: model lớn tốn **22,50 USD/ngày**, model nhỏ **2,25 USD/ngày**. Đây là minh họa theo giá đề bài, không phải hóa đơn API thực tế hoặc xác nhận giá hiện hành. Hàm `compare_models` chỉ ước tính phí **output** từ số từ theo template, nên không phải tổng phí request.

Model lớn có thể đáng chi phí cho câu hỏi kỹ thuật nhiều ràng buộc nếu kiểm thử cho thấy cải thiện chất lượng đủ lớn. Model nhỏ phù hợp phân loại ý định, định tuyến và FAQ đơn giản khi đã đạt yêu cầu; lab spam cũng cho thấy hai model zero-shot có thể cùng F1 trên một tập nhỏ. Việc dùng Groq miễn phí trong lab chính là tình huống riêng, không thay đổi phép tính giả định này.

## 2.3. Trải nghiệm người dùng với streaming

Streaming hữu ích khi người dùng đang chờ câu trả lời dài trong chatbot: họ có thể đọc phần đầu trong lúc model tiếp tục sinh, giảm cảm giác chờ đợi dù tổng thời gian chưa chắc ngắn hơn. Non-streaming phù hợp với phân loại spam, xử lý batch hoặc nhận JSON cần kiểm tra đầy đủ trước khi dùng. Chatbot trong bài in từng phần `delta.content`, ghép thành câu trả lời hoàn chỉnh và giữ ba lượt hỏi–đáp gần nhất cho request tiếp theo; chunk rỗng được bỏ qua.

## Ghi nhận sử dụng AI và kiểm thử

Chạy từ gốc repo: `.\.venv\Scripts\python -m pytest tests/test_solution.py -v` — **19 passed**. Kiểm tra mock bổ sung xác nhận chatbot giữ đúng ba cặp hỏi–đáp sau năm lượt, bỏ chunk rỗng và ghép phản hồi; retry chờ đúng 0,1 rồi 0,2 giây khi hai lần đầu lỗi.

Em dùng AI hỗ trợ triển khai bảy hàm từ template, giải thích retry/streaming, soạn phần phản ánh và chạy test mock. Các khóa `gpt4o_*`/`mini_*`, chữ ký hàm và bộ test gốc được giữ nguyên. Kết quả test mock chỉ xác nhận hành vi trong test, chưa xác nhận tài khoản, model hoặc phản hồi từ API thật.

Tham khảo cách đọc streaming và tham số request: [OpenAI streaming](https://developers.openai.com/api/docs/guides/streaming-responses), [Chat Completions Python reference](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create).
