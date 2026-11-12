# What I Studied in Each Subject and What I Built

> TEMPLATE: the subject descriptions are general. 

## Programming Fundamentals
Programming fundamentals is the first programming course in computer science. We learned variables, conditions, loops, functions, arrays and file handling, using C++. My project in this subject was a digital security locker written in C++. user has to enter correct username and password to get access, after 3 wrong attempts the locker would lock for 30 seconds.


## Object-Oriented Programming (OOP)
In OOP we studied classes, objects, encapsulation, inheritance and polymorphism, so that programs are organized around objects instead of long scripts. My project was a student management system. It has many features, such as adding, updating and searching students, and the language we used is C++.

## Digital Logic Design (DLD)
DLD explains how digital hardware works: logic gates, Boolean algebra, combinational circuits, flip-flops and state machines. My project was a smart environment control system. It detects the temperature of the environment and turns LEDs on according to temperature ranges I defined. It can also detect humidity. It uses sensors, LEDs and a buzzer. More details are in the smart environment document.

## Database Management Systems (DBMS)
In DBMS we studied the relational model, entity relationship diagrams, normalization, SQL queries, joins and transactions. My project was an online pet shop and vet system, built with SQL.It managed Admins, pets, products, customers, appointments, pet breed, orders etc.

## Computer Networks
In computer networks we studied the OSI and TCP/IP models, IP addressing, protocols, switching and routing, and network security basics. My project was an RFID intruder detection system. RFID uses a reader and tags that carry an identity, so a system can recognize who or what is present. 

## Data Science
Data science covers cleaning data, exploring it, building prediction models and checking how good they are. I built a flight delay prediction project using Python. 

## Deep Learning
Deep learning uses neural networks to learn patterns from data such as images, sound and text. My project was intruder classification and suspicious activity detection. 

## Cloud Computing
In cloud computing we studied virtual machines, storage, services, scalability and deploying applications online. I created a RAG for the topic of student career anxiety. RAG means retrieval-augmented generation: the system first retrieves relevant passages from documents and then a language model answers using them. More details are in the career anxiety RAG document.

## Mobile Application Development
In mobile application development we learned to build apps for phones. I created an app called Campus Marketplace using React Native, where students can buy and sell items on campus, or describe what it really does.

## Final Year Project
My final year project is SentinalGuard, which is still going on. SentinelGuard is an AI-assisted Zero Trust framework that checks every control command before it reaches a physical actuator, such as the pump in a smart irrigation system. It follows a PIP → PDP → PEP pipeline. The PIP, a FastAPI and MQTT backend, collects the sensor readings and packages them as evidence. A cross-sensor Reliability Layer, trained on normal data only, and an AI Risk Model (Random Forest or XGBoost) then score each command. The PDP decides whether to allow, restrict or block it, and the PEP on the ESP32 enforces that decision. It targets sensor spoofing, sensor replay, command replay and stealthy spoofing (test cases C0 to C4). So far we have prepared SWaT Stage 1 data (FIT101, LIT101, P101, MV101) as a time-ordered normal/early-attack/late-attack split to validate the method. The next step is to recalibrate and retrain on our own hardware readings, because SWaT’s values and dynamics differ from our small sensors.

## My favorite subjects
My favorite subjects are Information security and Cyber forensics becuase in these we studied how to make the information secure and how to do forensics after any cyber attack.
