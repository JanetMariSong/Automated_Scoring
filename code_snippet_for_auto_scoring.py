#https://teamkairos.ai:6112/backend-api/v2/autoScore/2022/AB
    def _autoScore(self,year,type):
        try:
            recursiveScoring(int(year),type)
            return "test"

            
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_next)
            return {
                '_action': '_autoScore',
                'success': False,
                "error": f"an error occurred {str(e)}"}, 400


#https://teamkairos.ai/chatgpt-math/getFile.php?folder=DB/Calculus/AP-pastpaper/FRQ/AB/2022&ext=tex
def recursiveScoring(yyyy,TT):
    problemList = get('https://teamkairos.ai/chatgpt-math/getFile.php', params={
                    'folder': 'DB/Calculus/AP-pastpaper/FRQ/%s/%d' % (TT,yyyy),
                    'ext':'tex'
                }).json()
    
    imageList = get('https://teamkairos.ai/chatgpt-math/getFile.php', params={
                    'folder': 'DB/Calculus/AP-pastpaper/FRQ/%s/%dStudent' % (TT,yyyy),
                    'ext':'png'
                }).json()
    

    scoreDataSet = {}
    for problemData in problemList:
        filename = getFilename(problemData)

        scoreDataSet[filename] = {
            "student":[],
            "Contents":problemList[problemData]["Contents"],
            "Solution":problemList[problemData]["Solution"],
            "Rubric":problemList[problemData]["Rubric"]
        }
        for imageData in imageList:
            if imageData.find(filename) != -1:
                scoreDataSet[filename]["student"].append(getParentFolderName(imageData))

     
    for filename in scoreDataSet:
        questionNum = filename
        subquestion = ""
        problemFile = filename.split("-")
        if len(problemFile) == 2:
            questionNum = problemFile[0]
            subquestion = problemFile[1]

        Contents = scoreDataSet[filename]["Contents"]
        Solution = scoreDataSet[filename]["Solution"]
        Rubric = scoreDataSet[filename]["Rubric"]
        
        if len(scoreDataSet[filename]["student"]) == 0:
            continue

        system_message_content = "The user will upload an image of a response to the provided rubric. Evaluate the response based strictly on the scoring guidelines provided, and clearly indicate awarded points along with detailed justifications."
        assistant_message_content = "I understand the instructions and will follow them in my evaluation."

        user_message_content =f"""
```problem
{Contents}
```
```solution
{Solution}
```
```rubric
{Rubric}
```
Scoring result must follow the structure below.
```structure
- 1st Sentence or Mathematical expression
  - Criteria 1-1 : earned point / point
  - Criteria 1-2 : earned point / point
...
  - Criteria 1-k : earned point / point
...
- n-th Sentence or Mathematical expression
  - Criteria n-1 : earned point / point
  - Criteria n-2 : earned point / point
...
  - Criteria n-x : earned point / point
- total points : earned out of full point
```
Make sure total point be correctly calculated."""
       


        for i in range(1,8):
            nowDB = f"DB_{i}"

            for student in scoreDataSet[filename]["student"]:
                imageURL = 'https://teamkairos.ai/chatgpt-math/DB/Calculus/AP-pastpaper/FRQ/%s/%dStudent/%s/%s.png' % (TT,yyyy,student,filename)
                sendData = {
                    "model":"o1",
                    "messages":[
                        {
                            "role": "system",
                            "content": system_message_content
                        },
                        {
                            "role": "assistant",
                            "content": assistant_message_content
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_message_content},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": imageURL,
                                    },
                                },
                            ],
                        }
                    ]
                }
                

                [db,cursor] = StartDB()

                sql = "SELECT * from %s WHERE Year=%d AND Test_Type='%s' AND Question='%s' AND Subquestion='%s' AND Student_ID='%s'" % (nowDB,yyyy,TT,questionNum,subquestion,student)
                cursor.execute(sql)
                res = cursor.fetchall()

                if len(res) >= 1:
                    print(f"{yyyy}, {TT}, DB_{i}, {filename}=({questionNum},{subquestion}), {student}, continue")
                    continue
                else:
                    print(f"{yyyy}, {TT}, DB_{i}, {filename}=({questionNum},{subquestion}), {student}")

                response = ""

                while response == "":
                    responseData = post(
                        url     = "https://api.openai.com/v1/chat/completions",
                        proxies = None,
                        headers = {
                            'Authorization': 'Bearer foo-bar'
                        }, 
                        json    = sendData,
                        stream  = False
                    ).json()

                    if 'choices' in responseData:
                        response = responseData['choices'][0]['message']['content']

                    if response == "":
                        print("Break Time")
                        time.sleep(60)


                sql = f"INSERT INTO {nowDB} VALUES(%d,'%s','%s','%s','%s','%s',%d,'%s')" % (yyyy,TT,questionNum,subquestion,student,addslashes(response),-1,"")
                cursor.execute(sql)

                EndDB(db)
                #print(sendData) 
                #print(response)
                #break
            #break
        #print(Contents)
        
    return