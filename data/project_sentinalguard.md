# SentinalGuard (Final Year Project, In Progress)



## What it is
SentinalGuard is my final year project (FYP) at the University of Management and Technology. It is still going on, so it is not finished yet. SentinelGuard is a security layer for a smart irrigation system. It checks every command, such as “turn the pump on”, before it reaches the hardware, and blocks it if it looks fake or unsafe. It uses AI together with a Zero Trust rule: never trust a command until it has been verified.

## The problem 
Smart irrigation systems send sensor readings and commands over the network, so an attacker can spoof a sensor, replay old readings, or replay old commands. A fake command can make the pump run at the wrong time, which wastes water or damages crops. Many existing systems only detect attacks, and few stop them before they reach the hardware.

## What it does
The system is a pipeline. An ESP32 sends sensor readings over MQTT to a FastAPI backend (the PIP), which collects them. Two AI parts then score each command, and a decision layer (the PDP) allows or blocks it. The ESP32 (the PEP) enforces that decision. I worked on the AI part, so the data, features and model steps below are the ones I did or planned.


## My role and my team
I am handeling the AI/ML brancg of the project. currently i am working on the dataset.

