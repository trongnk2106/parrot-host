from app.src.v1.gemma_trainer.gemma_trainer import gemma_trainer
from app.src.v1.schemas.base import GemmaTrainerRequest
if __name__=="__main__": 
    data = [
        'Trong cuộc sống, chúng ta thường gặp phải những thách thức và khó khăn, nhưng quan trọng nhất là cách chúng ta đối phó với chúng và học hỏi từ những trải nghiệm đó.',
'Hãy nhớ rằng mọi điều tích cực mà bạn đặt ra trong cuộc sống đều đáng giá để theo đuổi, và mỗi bước tiến trong hành trình của bạn là một bước gần hơn đến mục tiêu cuối cùng.',
'Đừng bao giờ từ bỏ ước mơ của bạn, vì những ước mơ là nguồn động viên và sức mạnh để bạn tiến xa hơn trong cuộc sống.',
'Hãy tận hưởng những khoảnh khắc đẹp trong cuộc sống và luôn luôn nhớ rằng cuộc sống là một cuộc hành trình, không phải là điểm đến.',
'Mỗi ngày là một cơ hội mới để bạn thể hiện bản thân và trải nghiệm những điều mới mẻ trong cuộc sống.',
'Đừng để những thử thách và khó khăn làm bạn chùn bước, hãy nhìn nhận chúng như là cơ hội để phát triển và trưởng thành.',
'Hãy sống mỗi ngày với lòng biết ơn và tận hưởng những điều tích cực xảy ra xung quanh bạn, vì cuộc sống là một món quà quý giá mà chúng ta đều được ban cho.',
'Đừng bao giờ ngừng mơ mộng và tưởng tượng về một tương lai tươi sáng, vì mơ ước là động lực để bạn tiến xa hơn trong cuộc sống.',
'Hãy đặt mục tiêu lớn và hạnh phúc cho bản thân, nhưng đừng quên tận hưởng những khoảnh khắc nhỏ bé và giá trị đích thực của cuộc sống.',
'Cuộc sống không phải là về điều gì bạn có, mà là về điều gì bạn trở thành. Hãy luôn cố gắng trở thành phiên bản tốt nhất của chính mình và sống một cuộc sống ý nghĩa và hạnh phúc.',
'Các nhà khoa học đang tiến hành nghiên cứu sâu rộng về tương tác giữa gen và môi trường để hiểu rõ hơn về cơ chế gây ra các bệnh ung thư và tìm ra phương pháp điều trị hiệu quả hơn.',
'Nghiên cứu về trí não con người đang mở ra cánh cửa cho việc phát triển công nghệ AI và robo-trí tuệ nhân tạo, mang lại những tiến bộ vượt bậc trong lĩnh vực tự động hóa và hỗ trợ thông minh.',
'Công nghệ CRISPR-Cas9 đã mở ra cánh cửa cho việc chỉnh sửa gen một cách chính xác và hiệu quả, mở ra tiềm năng mới trong điều trị gen và nghiên cứu về sinh học phân tử.',
'Nghiên cứu về vật liệu siêu dẻo đang làm thay đổi cách chúng ta sản xuất và sử dụng vật liệu trong công nghiệp, tạo ra những sản phẩm với khả năng chịu lực và đàn hồi cao hơn.',
'Các nhà nghiên cứu đang phát triển công nghệ nano để chẩn đoán sớm và điều trị các bệnh ung thư, mang lại hy vọng cho những người mắc bệnh có thể được điều trị hiệu quả và nhanh chóng hơn.',
'Nghiên cứu về lượng tử đang mở ra một thế giới mới của tính toán và truyền thông, tạo ra cơ hội mới trong việc phát triển máy tính lượng tử và các ứng dụng có liên quan.',
'Các nhà khoa học đang nghiên cứu về hiệu ứng năng lượng mặt trời để tìm ra các phương pháp hiệu quả hơn để thu thập và lưu trữ năng lượng từ mặt trời, đóng vai trò quan trọng trong việc giảm thiểu ô nhiễm môi trường.',
'Nghiên cứu về sinh học phân tử đang tìm ra những cơ chế mới của sự sống và phát triển của sinh vật, cung cấp thông tin quan trọng cho việc phát triển thuốc và liệu pháp mới.',
'Công nghệ blockchain đang được ứng dụng rộng rãi trong lĩnh vực y tế để bảo vệ dữ liệu bệnh nhân và cải thiện quản lý thông tin y tế.',
'Nghiên cứu về trí tuệ nhân tạo và học máy đang dẫn đầu trong việc phát triển các hệ thống tự động hóa và tự học, mở ra tiềm năng mới trong nhiều lĩnh vực ứng dụng từ tự động hóa công nghiệp đến xe tự lái và dịch vụ khách hàng.',
    ]
    request_data = {
    "task_id": "123",
    "data": data,
    "num_train_epochs": 10
    }

    result = gemma_trainer(
        celery_task_id="123",
        request_data=GemmaTrainerRequest(**request_data)
    )


