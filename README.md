# NOVEL CYBER ATTACK DETECTION USING A HYBRID DEEP LEARNING MODEL

## Abstract

A total of 5 billion people around the world use the internet nowadays – equivalent to 63 percent of the world's total population. Web clients proceed to develop as well, with the most recent information showing that the world's associated population developed by nearly 200 million in the one year to April 2022. The utilization and demand of the internet is increasing swiftly. Therefore, sensitive data is increasing day by day. As every minute passes by, the information created increases exponentially. The created information must be secure or it might lead to the divulgence of sensitive information of the users. The digital world connects everything and everybody to apps, data, purchases, services, and communication. The truth that nearly everybody in this world is currently more dependent on data and communication technology implies that for cybercriminals, there’s a booming criminal opportunity. So, not only the number of internet users are increasing day by day, the number of cybercriminals are also increasing with them. It seems there is no such protection for private data. When it comes to privacy, cyber security plays a crucial role here. Data breaches, Denial of Service attacks, Credential breaches are arduous to predict and businesses lose millions of dollars each day due to cyber attacks and leads to a bad reputation and credibility. As technology has developed, so has the dark web fortified its sophistication. It has provided a haven for cybercriminals and resulted in an increased threat on the surface Internet usage. These security threats have increased the importance of cyber security. Securing this world is essential for protecting people, organizations, habitats, and infrastructure. Cybercrime rate is increasing; hence, without cyber security, we could lose sensitive information, money, or reputation. Cyber security is as important as the current need for technology. No one is safe from the threat of cyber attacks in this digitalized world. If Cybercriminals can get to our computer, then they could easily steal our sensitive information. We need to possess some knowledge about the attacks if we want to stand a chance against these kinds of threats. For this reason, we have proposed the novel cyber attack detection system using a hybrid deep learning model which identifies the cyber attacks based on the various features extracted from the dataset (Kdd-cup’99) which contains four different attack classes. The cyber attack identification is done by a hybrid deep learning model which concatenates three individual algorithms for greater accuracy. CNN, MLP and LSTM are the three individual algorithms and then concatenated into a hybrid one which is used to detect and classify the type of attacks with greater accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Datasets](#datasets)
- [Credits](#credits)

## Introduction

This repository contains the code and documentation for the "NOVEL CYBER ATTACK DETECTION USING A HYBRID DEEP LEARNING MODEL" project. The aim of this project is to develop a novel cyber attack detection system using a hybrid deep learning model that can identify various types of cyber attacks with greater accuracy.

## Features

- Hybrid deep learning model combining CNN, MLP, and LSTM algorithms.
- Detection and classification of cyber attacks using the Kdd-cup’99 dataset.
- Predict cyber attack classes based on user-provided input.
- Utilizes a machine learning model trained on diverse attack datasets.
- Supports multiple attack categories for accurate classification.
- The web application is built using Flask, a lightweight and versatile Python web framework, making it accessible through a web browser. Users can input specific features related to network traffic and receive predictions about the type of attack detected.

## Datasets
- UNSW-NB15: 9 attack classes
- Kdd cup’99: 4 attack classes
- NSL-KDD: 24 attacks, 4 classes

A web application has been developed using Flask, which predicts attack classes based on user-provided features. The input features include:
- Attack Type
- Count
- Destination Host Different Service Rate
- Destination Host Same Source Port Rate
- Destination Host Same Service Rate
- Destination Host Server Count
- Flag
- Logged In
- Same Service Rate
- Serror Rate
- Protocol Type

The final prediction will fall into one of the following attack classes:
- Normal
- DOS (Denial of Service)
- Probe
- U2R (User to Root)
- R2L (Remote to Local)

## Credits

- Preethika N C
- Swetha Nachimuthu
- Subiksha T
