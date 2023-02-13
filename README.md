# Detecting_Suicidality_Contextual_GNN_CLPsych_2022

- [Daeun Lee](https://sites.google.com/view/daeun-lee/about?authuser=0), Mygeoung Kang, Minji Kim, and [Jinyoung Han](https://sites.google.com/site/jyhantop/)
- [Project page](https://sites.google.com/view/daeun-lee/dataset/CLPsych2022?authuser=0)


# Abstract
Discovering individuals' suicidality on social media has become increasingly important. Many researchers have studied to detect suicidality by using a suicide dictionary. However, while prior work focused on matching a word in a post with a suicide dictionary without considering contexts, little attention has been paid to how the word can be associated with the suicide-related context. To address this problem, we propose a suicidality detection model based on a graph neural network to grasp the dynamic semantic information of the suicide vocabulary by learning the relations between a given post and words. The extensive evaluation demonstrates that the proposed model achieves higher performance than the state-of-the-art methods. We believe the proposed model has great utility in identifying the suicidality of individuals and hence preventing individuals from potential suicide risks at an early stage.
	
# Creating a Suicide Dictionary
We create a word-level English suicide dictionary in a computational way using the UMD Reddit Suicidality Dataset (Shing et al., 2018; Zirikly et al., 2019). The dataset contains the assessment of the severity of suicidality of 866 Reddit users who had posted on the r/SuicideWatch subreddit from 2008 to 2015 and their 79,569 posts uploaded to 37,083 subreddits. The annotation of the suicidality levels (i.e., No risk, Low risk, Moderate risk, and Severe risk) was conducted by crowdsourcing and domain experts. We only use the posts uploaded to the r/SuicideWatch and 15 mental-health-related subreddits (e.g., r/depression, r/anxiety, r/selfharm, etc.) (Gaur et al., 2018)  as a target group and use the posts of users who had not posted on either r/SuicideWatch or mental-health related subreddits as a control group. 
Before constructing a dictionary, we anonymize the dataset by removing personally identifiable information such as names, email addresses, and URLs. After removing stopwords, and lemmatizing the text using spaCy (Honnibal and Montani,2017), we extract keywords for each post using KeyBERT (Grootendorst, 2020), and then apply the sparse additive generative model (SAGE) (Eisenstein et al., 2011) to determine the words specialized for each label compared to the entire lexicon. Finally, the constructed dictionary includes 297 suicide-related words. Note that the words belonging to the control group are excluded from the corpus set of each label.  
	
  
# Data Download
To download the dataset, please send email to delee12@skku.edu.
Suicide Dictionary (csv file) : 5.6KB

# Page URL
You can find the link to the research paper of this dataset and the lexicons [here](https://aclanthology.org/2022.clpsych-1.10/
).  If our work was helpful in your research, please kindly cite this work:
BIBTEX
@inproceedings{lee2022detecting,
  title={Detecting Suicidality with a Contextual Graph Neural Network},
  author={Lee, Daeun and Kang, Migyeong and Kim, Minji and Han, Jinyoung},
  booktitle={Proceedings of the Eighth Workshop on Computational Linguistics and Clinical Psychology},
  pages={116--125},
  year={2022}
}


# Acknowledgments
Acknowledgments
This research was supported by the Ministry of Education of the Republic of Korea and the National Research Foundation of Korea (NRF-2022S1A5A8054322), and the MSIT (Ministry of Science and ICT), Korea, under the ICAN (ICT Challenge and Advanced Network of HRD) program (IITP-2021-2020-0-01816) supervised by the IITP (Institute of Information & Communications Technology Planning & Evaluation).

# Our Lab Site
[Data Science & Artificial Intelligence Laboratory (DSAIL) @ Sungkyunkwan University](https://sites.google.com/view/datasciencelab)
