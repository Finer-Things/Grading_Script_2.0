from grading_functions import *
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC


def darken_color(color, amount=1.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> darken_color('g', 0.3)
    >> darken_color('#F034A3', 0.6)
    >> darken_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


class Grades:
    def __init__(self, course_name = None, quarter_name = None, grade_item_categories = None, grade_category_dictionary = None, gradescope_file_name = None, egrades_roster_file_name = None, webwork_file_name = None):
        self.course_name = course_name
        self.quarter_name = quarter_name
        self.grade_item_categories = grade_item_categories
        #This dictionary carries some helpful information to accompany each category. 
        #The entries in the tuple are: (grading_weight, normalization_status, number_to_be_dropped)
        #There could also be a curve function parameter added later
        self.grade_category_dictionary = grade_category_dictionary
        self.gradescope_file_name = gradescope_file_name
        self.egrades_roster_file_name = egrades_roster_file_name
        self.webwork_file_name = webwork_file_name
        self.grade_items_by_category = None
        self.gradescope_df = None
        self.egrades_roster_df = None
        self.webwork_df = None
        self.webwork_total_col_name = None
        #A counter for how many grade items each category has. Once the Grade Item Categories are set, there should be a method that creates
        #this dictionary with keys the Grade Item Categories and values their number of assignments that qualify. 
        #For example, if there are 2 midterms then this function should return 2 when you enter the "Midterm" category as a key. 
        self.num_grade_items_by_category = None
        #This attribute will come from the Gradescope dataframe and keeps track of the max points for each assignment by assignment name
        self.max_points_function = None
        #Grades is a generic atribute that defaults to the Gradescope dataframe and updates to the merged dataframe if a merge method is used
        self.grades = None
        self.final_condition = False
        self.great_effort_rule = False

    def set_grade_category_dictionary(self, category_dictionary = None, grade_item_categories = None):
        if grade_item_categories == None:
            grade_item_categories = self.grade_item_categories
        if category_dictionary == None:
            category_dictionary = self.grade_category_dictionary
        #Set a list of Grade Categories, such as Homework, Quiz, Midterm, Final
        self.grade_item_categories = grade_item_categories
        if category_dictionary == None or category_dictionary.keys != grade_item_categories:
            self.grade_category_dictionary = {category:(5, False, 0) for category in grade_item_categories}
        else:
            self.grade_category_dictionary = category_dictionary
    
    def get_gradescope_df(self, file_name = None, course_host = "Canvas"):
        if file_name == None:
            file_name = self.gradescope_file_name
        if file_name == None:
            raise Exception("Cannot upload Gradescope CSV file. No file name stored in this instance.")
        else:
            self.gradescope_df = pd.read_csv(file_name)
            if course_host == "Canvas":
                pass #Eventually, rename Student ID column or switch it so their netid matches. Maybe nothing nees to be done in this case. 
            elif course_host == "Gauchospace":
                pass #This should work as-is, but maybe it will need to be updated if we change what happens in the "Canvas" case. 

    def get_egrades_roster_df(self, file_name = None):
        if file_name == None:
            file_name = self.egrades_roster_file_name
        if file_name == None:
            raise Exception("Cannot upload Egrades CSV file. No file name stored in this instance.")
        else:
            self.egrades_roster_df = pd.read_csv(file_name)
            self.egrades_roster_df.rename(columns = {"Perm #": "SID"}, inplace = True)
            self.egrades_roster_df = self.egrades_roster_df[["SID", "Enrl Cd", "Email"]]

    def get_webwork_df(self, file_name = None, webwork_total_col_name = None, course_host = "Canvas"):
        if file_name == None:
            file_name = self.webwork_file_name
        if webwork_total_col_name == None:
            webwork_total_col_name = self.webwork_total_col_name
        if file_name == None:
            raise Exception("Cannot upload Webwork CSV file. No file name stored in this instance.")
        else:
            self.webwork_df = pd.read_csv(self.webwork_file_name)
        
        if course_host in [None, "Canvas"]:
            self.webwork_df.rename(columns={col_name:"SID" for col_name in self.webwork_df if "login ID" in col_name})
        
        #Renaming ID Columns
        if course_host == "Gauchospace":            
            #Renameing "Perm #" to "SID" in Egrades, then pruning it to almost nothing
            self.webwork_df.rename(columns = {webwork_total_col_name: "Homework", "ID number": "SID"}, inplace = True)
            self.webwork_df = self.webwork_df[["SID", "Homework"]]
            self.webwork_df["Homework"] = self.webwork_df.apply(lambda row: float(str(row["Homework"]).strip("%")), axis=1)
            self.webwork_df["Homework - Max Points"] = 100
        elif course_host in [None, "Canvas"]:
            self.webwork_df.columns = self.webwork_df.columns.str.strip()
            self.webwork_df.rename(columns = {self.webwork_df.columns[0]: "Email", r"%score": "Homework"}, inplace = True)
            self.webwork_df = self.webwork_df[["Email", "Homework"]]
            # The next command strips the spaces out of the columns and returns only the login ID (before the @ symbol in the email address)
            self.webwork_df["Email"] = self.webwork_df["Email"].apply(lambda entry: entry.strip().split("@")[0])
            self.webwork_df["Homework"] = self.webwork_df["Homework"].apply(lambda entry: float(entry)/.9)


        
    def get_all_grading_data(self, course_host = "Canvas", webwork_total_col_name = None):
        if course_host in [None, "Canvas"]:
            # This still merges files separately, NOT through Canvas. But the merging is on "Email", which I trunkated because egrades and canvas can't get the students' emails straight! 
            if self.gradescope_file_name != None:
                self.get_gradescope_df(course_host = course_host)
                self.grades = self.gradescope_df
                self.grades["Email"] = self.grades["Email"].apply(lambda entry: entry.split("@")[0])
                if self.egrades_roster_file_name != None:
                    self.get_egrades_roster_df()
                    self.egrades_roster_df["Email"] = self.egrades_roster_df["Email"].apply(lambda entry: entry.split("@")[0])
                    self.grades = pd.merge(self.gradescope_df, self.egrades_roster_df, on="Email", how ="right")
                if self.webwork_file_name != None:
                    self.get_webwork_df(course_host = course_host, webwork_total_col_name = webwork_total_col_name)
                    self.grades = pd.merge(self.grades, self.webwork_df, on="Email", how ="left")
        
        if course_host == "Gauchospace":
            if self.gradescope_file_name != None:
                self.get_gradescope_df(course_host = course_host)
                self.grades = self.gradescope_df
                if self.egrades_roster_file_name != None:
                    self.get_egrades_roster_df()
                    self.grades = pd.merge(self.gradescope_df, self.egrades_roster_df, on="SID", how ="right")
                    if self.webwork_file_name != None:
                        self.get_webwork_df(course_host = course_host, webwork_total_col_name = webwork_total_col_name)
                        self.grades = pd.merge(self.grades, self.webwork_df, on="SID", how ="left")
        
        self.set_max_points()
        self.set_grade_items_by_category()
        self.set_grade_category_dictionary
        self.set_num_grade_items_by_category()
        self.drop_junk_columns()
            
                    

    def set_grading_data(self):
        self.create_total_columns()
        self.create_grade_columns()
        #Create Letter Grades Columns
    
    def get_and_set_grades(self, course_host = "Canvas", webwork_total_col_name = None):
        self.get_all_grading_data(course_host = course_host, webwork_total_col_name = webwork_total_col_name)
        self.set_grading_data()
    
    def set_grade_items_by_category(self, grade_item_categories = None):
        if grade_item_categories == None:
            grade_item_categories = self.grade_item_categories
        
        try:
            junk_column_indices = [i for i, column in enumerate(self.grades.columns) if "Max Points" in column or "Submission Time" in column or "Lateness" in column]
            non_redundant_columns = [column for column in self.grades.columns if column not in self.grades.columns[junk_column_indices]]
            self.grade_items_by_category = {grade_category:[name for name in non_redundant_columns if grade_category in name] for grade_category in grade_item_categories}
        except:
            print("Could not compute grade_items_by_category because either gradescope_df or grade_item_categories is likely still set to None.")
            

    def set_num_grade_items_by_category(self, grade_items_by_category = None):
        if grade_items_by_category == None:
            grade_items_by_category = self.grade_items_by_category
        try:
            self.num_grade_items_by_category = {grade_item_category: len(grade_items_by_category[grade_item_category]) for grade_item_category in self.grade_item_categories}
        except:
            print("Cannot set attribute num_grade_items_by_category because either grade_item_categories or grade_items_by_category is still set to None.")
        
    def set_max_points(self, df = None):
        if df == None:
            df = self.grades
        max_points_column_names = [name for name in df.columns if "Max Points" in name]
        part_to_chop = len(" - Max Points")
        self.max_points_function = {name[:-part_to_chop]: df[name].iloc[1] for name in max_points_column_names}
        if self.webwork_file_name != None:
            self.max_points_function["Homework"] = 100
    
    def create_total_columns(self, category_dictionary = None, df = None):
        if category_dictionary == None:
            self.grade_category_dictionary
        if df == None:
            df = self.grades
        ##Check to make sure every grading category that has assignments to be dropped also has its assignments normalized.
        checklist = [category for (category, tuple) in self.grade_category_dictionary.items() if tuple[1] == False and tuple[2] != 0]
        if checklist != []:
            raise Exception(f"This is a list of grade items with lowest items dropped that have not been normalized. \n {checklist} \n It should be empty.")
        for (grade_category, tuple) in self.grade_category_dictionary.items():
            normalize_condition = tuple[1]
            d = tuple[2]# number of items dropped
            if normalize_condition:
                for grade_item_name in self.grade_items_by_category[grade_category]:
                    df[grade_item_name] *= 100/self.max_points_function[grade_item_name]
                    self.max_points_function[grade_item_name] = 100

            # Control flow below ensures empty categories have a zero total. When grade columns are created, these will not be factored into grade calculations. 
            if self.num_grade_items_by_category[grade_category] > 0:
                # What I've used to make this work without issue is normalizing everything before I drop the lowest. See the grade_categories_to_normalize list above. 
                df[f"{grade_category} Total"] = df[self.grade_items_by_category[grade_category]].apply(lambda row: sum(sorted(row.fillna(0))[d:]), axis=1)
                print()
                self.max_points_function[f"{grade_category} Total"] = sum(sorted([self.max_points_function[item] for item in self.grade_items_by_category[grade_category]])[d:])
                
                # Normalizing the Totals
                df[f"{grade_category} Total"] *= 100/self.max_points_function[f"{grade_category} Total"]
                self.max_points_function[f"{grade_category} Total"] = 100
                #Rounding to Two Decimal Places
                df[f"{grade_category} Total"] = df[f"{grade_category} Total"].apply(np.round, decimals=2)
            else:
                df[f"{grade_category} Total"] = 0
            
        
    def create_grade_columns(self, great_effort_rule = None, final_condition = None):
        if great_effort_rule == None:
            great_effort_rule = self.great_effort_rule
        if final_condition == None:
            final_condition = self.final_condition

        def grade_calculator(row):
            # for (category, data) in self.grade_category_dictionary.items():
            #     if self.num_grade_items_by_category[category] > 0:
            #         print(category, data, data[0])
            return sum([row[category + " Total"]*data[0] for (category, data) in self.grade_category_dictionary.items()])/sum([data[0] for (category, data) in self.grade_category_dictionary.items() if self.num_grade_items_by_category[category] > 0])
            
        self.grades["Grade"] = self.grades.apply(grade_calculator , axis=1).apply(np.round, decimals=2)
        self.grades["Letter Grade"] = self.grades.apply(lambda row: letter_grade_assigner(row, final_condition = final_condition, great_effort_rule = great_effort_rule), axis=1)
        self.max_points_function["Grade"] = 100

    def letter_grade_assigner(row, final_condition = None, great_effort_rule = False, grade_column_name = "Grade", final_grade_column_name = "Final Total"):
        """
        This letter grade assigner takes a raw grade percentage and outputs a letter grade. There are two peculiarities to this function that are not in a standard letter grade assigner: 
        1) Math 34A/B have a tradition of grading with a "Great Effort Rule" that assigns a B to students who earn a C or better through raw score if they took most of the quizzes, came to lecture and completed at 
        least most of the homework. I took a quick look at the spreadsheet and did not find any students who fell into this category without having reasonably high homework and quiz scores, so I just applied this to 
        everyone. 
        2) This was a tough quarter, and some students struggled all quarter with low grades and ended up doing a stellar job on the final. I believe that these students earned the grade they got on the final 
        (especially because this was not a take-home exam, which would offer them the chance to get "helped" by someone else). So the students who received a higher percentage on the final than they did in the class 
        were graded according to their final score. The great effort rule was not applied in this case, nor should it be. 
        Functionality: The variable num is used as a default for a letter grade assignment. You will see this not used in the case of being assigned B grades because the Great Effort Rule is based on their raw score. For the lines below, num could have been used instead of row["Final"] without change, but I think this could have easily muddied what was happening so I kept the "Final" argument. 
        """
        if final_condition == None:
            final_condition = self.final_condition
        
        if final_condition:
            num = max(row[grade_column_name], row[final_grade_column_name])
        else: 
            num = row[grade_column_name]
        if num > 104:
            return "huh?"
        elif num >= 97:
            return "A+"
        elif num >= 92.5:
            return "A"
        elif num >= 90:
            return "A-"
        elif num >= 87:
            return "B+"
        elif great_effort_rule and row[grade_column_name] >= 72.5: #For Math 34A/B's "Great Effort Rule" - never based on higher final grade!
            return "B"
        elif num >= 82.5:
            return "B"
        elif num >= 80:
            return "B-"
        elif num >= 77:
            return "C+"
        elif num >= 72.5:
            return "C"
        elif num >= 70:
            return "C-"
        elif num >= 67:
            return "D+"
        elif num >= 62.5:
            return "D"
        elif num >= 60:
            return "D-"
        else: 
            return "F"

    
    def display_point_totals_by_category(self):
        for grade_item_type in self.grade_item_categories: 
            print("\n", f"{grade_item_type}s  ({self.num_grade_items_by_category[grade_item_type]} total)", sep="")
            print(f"Index \t Pts \t Name")
            for index, col_name in enumerate(self.grade_items_by_category[grade_item_type]):
                print(f"{index+1} \t {self.max_points_function[col_name]} \t {col_name}")

    def drop_junk_columns(self, df = None):
        if df == None:
            df = self.grades
        junk_column_indices = [i for i, column in enumerate(df.columns) \
                       if "Max Points" in column or "Submission Time" in column \
                       or "Lateness" in column or column == "section_name" 
                       #or column == "Email"
                      ]
        df.drop(df.columns[junk_column_indices], axis=1, inplace=True)

    def stat_label(self, list, median_line_color = "", mean_line_color = ""):
        #used for plotting functions -- the stats are printed on the bottom
        return f"Size: {list[0]}, Mean ({mean_line_color}): {list[2]}, Median ({median_line_color}): {list[1]}, Std: {list[3]}, Min: {list[4]}, Max: {list[5]}"

    
    def print_student_grade_breakdown(self, student_name, position = None, use_last_name = False):
        if use_last_name == True:
            first_or_last_name = "Last Name"
        else:
            first_or_last_name = "First Name"
        display_list = ["First Name", "Last Name", "Grade"]+[grade_category+" Total" for grade_category in self.grade_item_categories]+["Letter Grade"]
        print_dataframe = self.grades[self.grades[first_or_last_name].apply(lambda entry: student_name.lower() in entry.lower())][display_list]
        if position == None:
            for category in display_list:
                print(f"{str(category)[:10]:13}|", end="")
            print("")
            for row in print_dataframe.itertuples():
                for index, category in enumerate(display_list):
                    print(f"{str(row[index+1])[:13]:13}|", end="")
                print("")
        else:
            print_list = print_dataframe.iloc[position]
            for item in display_list:
                print(f"{str(item)[:13]:13}|", end="")
            print("")
            for item in print_list:
                print(f"{str(item)[:10]:13}|", end="")
            print("")
    
    
    def plot_letter_grades(self, 
                           df=None, 
                           grade_category_to_separate_by = None,
                           savefig = False
                    ):
        if df == None:
            df=self.grades
        df = df.copy()
        df["Letter Grade Letter"] = df["Letter Grade"].apply(lambda string: string.strip("+-"))
        df["Letter Grade +/-"] = df["Letter Grade"].apply(lambda string: string.strip("ABCDF"))
        if grade_category_to_separate_by != None:
            df["Letter Grade +/-"] = df.apply(lambda row: f"No {grade_category_to_separate_by}" if (all([math.isnan(row[assignment]) for assignment in self.grade_items_by_category[grade_category_to_separate_by]])) else row["Letter Grade +/-"], axis = 1)
            # Scratch Work for modifying the code above:
            #### We want the "apply" condition above to be something like math.isnan applied to every entry in self.grade_items_by_category[grade_category_to_separate_by]
            # all([math.isnan(row[assignment]) for assignment in self.grade_items_by_category[grade_category_to_separate_by]])
        df = df.sort_values("Letter Grade Letter", ascending = False)

        plt.figure(figsize = (12,6))
        mpl.rcParams['text.color'] = "navy"
        sns.histplot(data = df, 
                x = "Letter Grade Letter", 
                hue = "Letter Grade +/-", 
                hue_order = ["-", "", "+", f"No {grade_category_to_separate_by}"], 
                palette = ["r", "b", "gold", "k"], 
                multiple="stack"
                ).set(title = f"{self.quarter_name} {self.course_name} Letter Grades" 
                                            # xlabel = f'nm = No {grade_category_to_separate_by}(s) Taken'
                                            )
        if savefig == True:
            plt.savefig(f"images/{self.quarter_name} {self.course_name} Letter Grades.png", bbox_inches = "tight")
        plt.show()
        plt.close()
    
    
    def show_me_the_colors(self, include_xkcd = False):
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors


        def plot_colortable(colors, title, sort_colors=True, emptycols=0):

            cell_width = 212
            cell_height = 22
            swatch_width = 48
            margin = 12
            topmargin = 40

            # Sort colors by hue, saturation, value and name.
            if sort_colors is True:
                by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                                name)
                                for name, color in colors.items())
                names = [name for hsv, name in by_hsv]
            else:
                names = list(colors)

            n = len(names)
            ncols = 4 - emptycols
            nrows = n // ncols + int(n % ncols > 0)

            width = cell_width * 4 + 2 * margin
            height = cell_height * nrows + margin + topmargin
            dpi = 72

            fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
            fig.subplots_adjust(margin/width, margin/height,
                                (width-margin)/width, (height-topmargin)/height)
            ax.set_xlim(0, cell_width * 4)
            ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.set_axis_off()
            ax.set_title(title, fontsize=24, loc="left", pad=10)

            for i, name in enumerate(names):
                row = i % nrows
                col = i // nrows
                y = row * cell_height

                swatch_start_x = cell_width * col
                text_pos_x = cell_width * col + swatch_width + 7

                ax.text(text_pos_x, y, name, fontsize=14,
                        horizontalalignment='left',
                        verticalalignment='center')

                ax.add_patch(
                    Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                            height=18, facecolor=colors[name])
                )

            return fig

        plot_colortable(mcolors.BASE_COLORS, "Base Colors",
                        sort_colors=False, emptycols=1)
        plot_colortable(mcolors.TABLEAU_COLORS, "Tableau Palette",
                        sort_colors=False, emptycols=2)

        plot_colortable(mcolors.CSS4_COLORS, "CSS Colors")

        # Optionally plot the XKCD colors (Caution: will produce large figure)
        if include_xkcd == True:
            xkcd_fig = plot_colortable(mcolors.XKCD_COLORS, "XKCD Colors")
        #xkcd_fig.savefig("XKCD_Colors.png")

        plt.show()


    def plot_grade_item(self, 
                    grade_item, 
                    df=None, 
                    max_score = 100, 
                    auto_max_score = True, 
                    stat = "count", 
                    savefig = False, 
                    graph_color="mediumpurple", 
                    over_achiever_color = "rebeccapurple",
                    mean_line_color="darkred", 
                    mean_line_width = 1.5,
                    mean_line_alpha=.95, 
                    median_line_color="mediumblue", 
                    median_line_width = 2,
                    median_line_alpha=.95, 
                    show_plot = True
                   ):
        if df == None:
            df=self.grades
        mpl.rcParams['text.color'] = graph_color
        plt.figure(figsize = (17,6))
        plt.rcParams["axes.titlesize"] = 30 
        col = df[df[grade_item].notna()][grade_item]
        stat_list = [col.count(), np.round(col.median(), decimals=1), np.round(col.mean(), decimals=1), np.round(col.std(), decimals=1), np.round(col.min(), decimals=1), np.round(col.max(), decimals=1)]
        
        #Grade Item Max Score -- Derived from the Max Points Function
        if auto_max_score == True:
            max_score = self.max_points_function[grade_item]

        sns.histplot(col, 
                    kde=True, 
                    bins=[max_score/10*n for n in range(11)], 
                    stat=stat, 
                    #ax=axes[0], 
                    color=graph_color).set(title = f"{grade_item.capitalize()}", 
                                            xlabel = self.stat_label(stat_list, median_line_color = median_line_color, mean_line_color = mean_line_color)
                                            )
        if stat_list[-1] > max_score:
            sns.histplot(df[max_score < df[grade_item]][grade_item], 
                        bins=[max_score, stat_list[-1]], 
                        color=over_achiever_color
                        )
        
        #Dashed lines for means%
        plt.axvline(df[grade_item].mean(), linestyle = "dashdot", linewidth = mean_line_width, color = mean_line_color, alpha = mean_line_alpha)


        #Dotted Lines for Medians
        plt.axvline(df[grade_item].median(), linestyle = "dashed", linewidth = median_line_width, color = median_line_color, alpha = median_line_alpha)




        if savefig == True:
            plt.savefig(f"{self.course_name}/images/{self.quarter_name} {self.course_name} {grade_item} Distribution.png", bbox_inches = "tight")
        
        if show_plot == True:
            plt.show()
            plt.close()
    
    def plot_by_type(self, grade_item_type, 
                        df = None, 
                        max_score = 100, 
                        auto_max_score = True, 
                        stat = "count", 
                        savefig = False, 
                        graph_color="mediumpurple", 
                        darken_color_factor = 1.5,
                        over_achiever_color = "rebeccapurple",
                        mean_line_color = "darkred", 
                        mean_line_alpha = .95, 
                        mean_line_width = 1.5,
                        median_line_color = "mediumblue", 
                        median_line_alpha = .95, 
                        median_line_width = 2
                    ):
        if df == None:
            df = self.grades
        #Style Setting
        mpl.style.context("fivethirtyeight", after_reset=False)
        
        #Grade Item List Length
        grade_item_list_length = self.num_grade_items_by_category[grade_item_type]
        
        #Figure Name
        plt.figure(figsize = (17,3+5*(grade_item_list_length+1)))
        #fig, axes = plt.subplots(2, 3, figsize=(22, 8)) #The axes argument didn't work the way I wanted it to with enlarged subplots
        plt.subplots_adjust(hspace=0.3, wspace = .2)
        # plt.suptitle(fig_title, fontsize=50, pad=20)
        fig_title = f"All {grade_item_type.capitalize()} Distributions"
        plt.figtext(.515,.93 - .004*grade_item_list_length,fig_title, fontsize=40, ha='center', color=darken_color(over_achiever_color, 1.07)) 
        """Creates a title without 
        a huge gap. Old title code commented out above this line.""" 
        plt.rcParams["axes.titlesize"] = 30 #makes the title size bigger for each plot

        for i, grade_item in enumerate(self.grade_items_by_category[grade_item_type]):
            plt.subplots_adjust(hspace=0.3, wspace = .2)

            mpl.rcParams['text.color'] = graph_color
            
            #Grade Item Max Score -- Derived from the Max Points Function
            if auto_max_score == True:
                max_score = self.max_points_function[grade_item]

            ##Hist Plot
            plt.subplot(grade_item_list_length+1,1,1 + i)
            col = df[grade_item]
            stat_list = [col.count(), col.median(), np.round(col.mean()), np.round(col.std()), np.round(col.min()), np.round(col.max())]
            sns.histplot(col, 
                        kde=True, 
                        bins=[max_score/10*n for n in range(11)], 
                        stat=stat, 
                        #ax=axes[0], 
                        color=graph_color).set(title = f"{i+1}) {grade_item}", 
                                                xlabel = self.stat_label(stat_list, median_line_color = median_line_color, mean_line_color = mean_line_color)
                                                )

            if stat_list[-1] > max_score:
                sns.histplot(df[max_score < df[grade_item]][grade_item], 
                            bins=[max_score, stat_list[-1]], 
                            color=over_achiever_color
                            )

            #Dashed lines for means%
            plt.axvline(df[grade_item].mean(), linestyle = "dashdot", linewidth = mean_line_width, color = mean_line_color, alpha = mean_line_alpha)


            #Dotted Lines for Medians
            plt.axvline(df[grade_item].median(), linestyle = "dashed", linewidth = median_line_width, color = median_line_color, alpha = median_line_alpha)


        #Plotting the Total Graph
        plt.subplot(grade_item_list_length+1,1,grade_item_list_length+1)
        grade_item = grade_item_type + " Total"
        col = df[df[grade_item] > 0][grade_item]
        stat_list = [col.count(), col.median(), np.round(col.mean()), np.round(col.std()), np.round(col.min()), np.round(col.max())]
        #Grade Item Max Score -- Derived from the Max Points Function
        sns.histplot(col, 
                    kde=True, 
                    bins=[10*n for n in range(11)],
                    stat=stat, 
                    #ax=axes[0], 
                    color=darken_color(graph_color, darken_color_factor)).set(title = f"{grade_item_type} Total", 
                                            xlabel = self.stat_label(stat_list, median_line_color = median_line_color, mean_line_color = mean_line_color)
                                            )

        if stat_list[-1] > 100:
            sns.histplot(df[100 < df[grade_item]][grade_item], 
                        bins=[100, stat_list[-1]], 
                        color=darken_color(over_achiever_color,1.2)
                        )

        #Dashdot Line for means%
        plt.axvline(df[grade_item].mean(), linestyle = "dashdot", linewidth = mean_line_width, color = darken_color(mean_line_color, .95), alpha = median_line_alpha)


        #Dashed Line for Medians
        plt.axvline(df[grade_item].median(), linestyle = "dashed", linewidth = median_line_width, color = darken_color(median_line_color, .95), alpha = median_line_alpha)

        
        if savefig == True:
            plt.savefig(f"images/All {self.quarter_name} {self.course_name} {grade_item_type} Distributions.png", bbox_inches = "tight")
        plt.show()
        plt.close()




    def scatter_plot(self, grade_items, against_items = ["Grade"], test_grade_item = "Final Total", palette = None, palette_list = None, num_columns = 3):
        self.grades[f"Took {test_grade_item}"] = self.grades.apply(lambda row: row[test_grade_item] > 0, axis=1)
        
        palette_1 = {True:"darkgreen",
                False:"lightcoral"}

        # palette_2 = {True:"midnightblue",
        #         False:"lightcoral"}

        if palette == None:
            palette = palette_1

        if palette_list == None:
            palette_list = [palette_1]
        
        p_len = len(palette_list)

        def div_up(a, b): # cd = "ceiling division"
            return -(a // -b)
        
        pair_list = [(grade_item, against_item) for grade_item in grade_items for against_item in against_items]
        num_pairs = len(pair_list)
        num_rows = div_up(num_pairs, num_columns)

        sns.set()
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(6.5*num_columns, 6.5*num_rows))
        
        fig_title = f"Grade Item Comparison Chart"
        plt.figtext(.515,.91 - .004*num_rows,fig_title, fontsize=40, ha='center', color="midnightblue") 
        """Creates a title without 
        a huge gap. Old title code commented out above this line.""" 
        plt.rcParams["axes.titlesize"] = 30 #makes the title size bigger for each plot

        for i, (grade_item, against_item) in enumerate(pair_list):
            sns.scatterplot(data=self.grades, 
            x=grade_item, y=against_item, 
            ax=axes[i//num_columns, i%num_columns],
            palette=palette_list[(i//len(against_items))%p_len], 
            hue = f"Took {test_grade_item}"
            )
        # plt.suptitle("Scatter Plots")
        # plt.savefig("Images\Features vs Grade Scatter Plots.png")
        # plt.tight_layout()
        plt.show()
        plt.clf()







        # self.grades.drop(self.grades.columns["Took {test_grade_item}"], axis=1, inplace=True)

    def plot_linear_test(self, grade_item_1, grade_item_2, test_grade_item = "Final Total"):
        gfs = self.grades[self.grades[test_grade_item] > 0]
        
        # index_names = gfs[ gfs[test_grade_item] == 0 ].index
        # gfs.drop(index_names, inplace = True)

        gfs = gfs[[grade_item_1, grade_item_2]].dropna()

        features = np.array(gfs[[grade_item_1]])
        #features = features.reshape(-1,1)
        outcome = gfs[grade_item_2]

        #print(features)

        features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, 
                                                                                    train_size = .8, 
                                                                                    random_state = 805)

        model = LinearRegression()
        model.fit(features_train, outcome_train)

        print(model.score(features_test, outcome_test))

        prediction = model.predict(features)

        error_values = [y_2 - y_1 for y_2, y_1 in zip(outcome, prediction)]


        sns.set()

        fig, axes = plt.subplots(1, 3, figsize=(25, 8))

        #Linear Regression vs Data
        plt.subplot(1,3,1)
        plt.scatter(features_test, outcome_test)
        plt.scatter(features_test, [model.coef_*x + model.intercept_ for x in features_test], alpha=.4)
        plt.title("Linear Regression vs Data")
        plt.xlabel("Midterm Total Adjusted")
        plt.ylabel("Grade")

        #plt.savefig("Images\Midterm Adjusted Versus Grade Linear Regression.png")

        #Fit Check for Linear Regression (Does this look like noise?)
        plt.subplot(1,3,2)
        plt.scatter(features, error_values)
        plt.plot([0,100], [0,0], color="g")
        plt.title("Visual Homoscedasticity Check")
        plt.xlabel("Midterm Total Adjusted")
        plt.ylabel("Error")

        plt.subplot(1,3,3)
        plt.hist(error_values, bins = 10)
        plt.title("Visual Normality Check")
        plt.xlabel("Error Values")
        plt.ylabel("Quantity")

        plt.show()







