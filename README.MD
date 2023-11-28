Introduction
============
This assessment is intended to give insights in your coding and logical thinking skills.

You are also asked to present your results and findings on a laptop. We can provide a laptop if needed. After your presentation we will have a discussion about these results, which considerations did you take into account, the issues you encountered, ... .

It is sufficient that you only finish the exercise as we have requested. However, feel free to spend as much time as you would like. There are no time restrictions that apply. Feel free to impress us with whatever you would like to share or just finish it as we requested.  Any insights in terms of refactoring or refinements can be explained when you come over to demonstrate.

Please treat this exercise as confidential so that we can reuse it for any future candidates. Before starting your assignment, please run `git init && git add . && git commit -am "initial commit"` so that we can track your changes from the base line and once you have finished, send us the archive of the local git repo.

If you have any questions at all, do not hesitate to contact us.

Good luck and enjoy!

Problem
============
Just about a couple of months ago, your colleagues managed to create artificially intelligent chat bots appeared to be chatting to each other in a strange language only they understood. The two chatbots came to create their own changes to English that made it easier for them to work – but which remained mysterious to the humans.

The robots had been instructed to work out how to negotiate between themselves, but they were not told to use comprehensible English, allowing them to create their own "shorthand" version of the language, which by first observations has quite simple grammar and consist of only three rules:

1. alphabet consists of 33 characters: 26 upper case letters and 7 digits: `ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567`
2. sentence always consists of 9 characters from the chat bot alphabet, never less and never more
3. chat bots never pronounce the same sentence more than once, hence there are no duplicates in their conversation

Here are a couple of examples of chat bot sentences collected by your colleagues:

 - RSWW1KS2S
 - 6LZTVD111
 - 6M2MMMMMM
 - XKKKKMZTK
 - SM2SKKZEE

As you can see from the above examples, their sentence structure is quite simple, every sentence consists of only 9 characters from chat bot alphabet. But the problem is that the meaning of these sentences is still unknown to humanity, and in order to reverse engineer chat bot language grammar, we simply need more data. To get that data, we decided to continuously record their conversations.

Unfortunately for us, the bots realized what we were up to, so they started to hide the last character. They are now replacing it with the question mark symbol "?". Here is what the conversations looks like now:

 - R2SR2SR2`?`
 - 2222MS22`?`
 - M22M22MS`?`
 - 62MSRSSS`?`
 - 2WKM1S2S`?`

As we all know, this kind of behavior on the robot part could prove to be dangerous. Therefore it is imperative we learn their language! You have been chosen to help us do this! It is crucial we design a machine learning algorithm that will help us recover those hidden last characters.

Assignment
==========
Your are given a couple of files:
 - `train.csv`, which contains clean and complete `2543` chat bot sentences collected before the chat bots had realized we were tapping their conversations
 - `hidden.csv`, where for each `379` chat bot sentence the 9th character is hidden to make it harder for humans to analyze chat bot language

You are also given the answers in the `answers.csv` data file, which you can use in order to estimate the accuracy of your intelligent data recovery algorithm.

Its worth to mention that all sentences in both `train.csv` and `hidden.csv` are unique, as it seems that chat bots never use a sentence more than once.

To reiterate, your task is to:
1. design and train machine learning algorithm from `train.csv` file to capture invisible patterns in the chat bot language
2. recover corrupted sentences from `hidden.csv` file with intelligent agent trained in the previous step
3. benchmark data correction accuracy by comparing recovered sentences with the sentences given in the `answers.csv` data file

Evaluation  criteria
====================
You can use any language and framework in order to solve the problem. If you are stuck or have any questions at all, do not hesitate to contact us, we will give you a hint.
