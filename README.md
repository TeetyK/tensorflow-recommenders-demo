[หลักการ]
datasets จาก tensorflow_recommenders <br>
1.UserModel ปรับ feature ให้เป็นพฤติกรรม คุณลักษณะ ให้เป็น Vector
  MovieModel ปรับ feature ให้เป็นพฤติกรรม คุณลักษณะ ให้เป็น Vector
  เพื่อแยกให้ แต่ละ vector ของเรื่องทีั่ชอบมาอยู่ใกล้กันมากที่สุด <br>
2.ค่า weight มาจาก ค่า Dot Product ที่ User และ Movie ใกล้เคียงกันมากที่สุด
ค่า Loss จะใช้ Factorized Top-K Cross Entropy ว่า User ที่คนนี้ดูอยู่อันดับต้นๆไหม <br>
3.Evaluation Metrix
Top-100 Categorical Accuracy ถ้าเราแนะนำ หนัง 100 เรื่อง มีเรื่องที่ดูจริงๆในนั้นกี่เปอร์เซ็นต์ และ Top100 โดยสุ่มข้อมูล rating <br>
4.Predict
ุถ้าเป็น User 42 ระบบจะแนะนำเรื่องอะไรให้เขาบ้าง