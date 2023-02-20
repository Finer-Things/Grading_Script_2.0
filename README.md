# Grading_Script_2.0
## Purpose
This script serves to provide an "all in one" functionality of my several previous grading scripts: 
* Uploading grades from up to three different CSV files, merging them and deleting Gradescope's 75% bloat columns that infest any upload of student grades from their site. 
* Computing Totals by category (such as Homework, Quizzes, Midterms) and sometimes dropping lowest assignments
* Computing a grade for each student based on the above totals and a letter grade that corresponds to that percentage, then saving this so grades can be submitted
* Creating histrograms with kde plots and basic statistics for either a single grade item or for an entire grading category. These have LOTS of keywords and you can tweak many things like the colors, the number of max points, etc. 
* Creating scatter plots to look for correlations
* Some linear regression and visual checks for the appropriateness of linear regression between two columns
* A couple other quality of life commands, such as quickly seeing a student's grade breakdown for the frequent grade-related questions from students around finals week
## Implementation
* The previous versions were very long, and the fact that they were procedural made them difficult to change for courses -- even with the modularity of the most recent setups. 
* This version has a class called Grades in its own file (roughly 700 lines of code) that not only hides much of the code but also sllows for multiple courses to be analyzed simultaneously. If you open the notebook file, you can see the instantiation and method calls. Yay OOP! 
## Planned Future Changes
* Because the campus is switching to Canvas, there will need to be more canvas functionality built into the methods of the Grades class. 
* More method calls for the machine learning that has been done in previous iterations. Linear regression isn't very strong for these features and there is not yet a normalization practice for the various models. I would like to implement these next. 
