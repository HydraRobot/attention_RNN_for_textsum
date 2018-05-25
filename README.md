# attention_RNN_for_textsum
This is to reproduce article &lt;A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION>
## Dataset: CNN_DailyMail ##
### 1.1 General Information: ###
This dataset has nearly 290 articles and 1260K story-question pairs totally: 
	90K of of CNN news and 380K of story-question pairs
	200K new stories and 880K story-question pairs
A news article is usually associated with a few (e.g., 3–5) bullet points, which is story-question, and each of them highlights one aspect of its content. 
The original question format is cloze-style. These questions are actually highlights of news, so the information of question is guaranteed to be included by the given story. In order to use the dataset in text sum task, questions are converted to text summary.

The vocabulary of this dataset is CNN 120K and DM 210K. It is very large vocabulary.

The passages are around 30 sentences and 800 tokens on average, while each question contains around 12–14 tokens. 
It is not very difficulty. Major is paraphrase task.

### 1.2 Dataset Process: ###

Input dataset files format is sequential binary data.
 Step 1: make example queue from input bin file. 
	1. Read the bin file as string to text format.   
	2. Process  input text into example. 
	3. Place the examples into the example queue. (Example queue is a queue having all input samples included.) 

Step2: create batch queue from examples in example queue.
	1. Take the examples out of the example queue.  
	2. Sort them by encoder sequence length
	3. Process it into batches and place them in the batch queue. 

Each time, get next batch from batch queue feed into model. 
