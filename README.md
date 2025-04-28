# pstat131-homework-5-solved
**TO GET THIS SOLUTION VISIT:** [PSTAT131 Homework 5 Solved](https://www.ankitcodinghub.com/product/pstat-131-homework-5-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;119134&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;PSTAT131 Homework 5 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
For this assignment, we will be working with the file “pokemon.csv”, found in /data. The file is from Kaggle: https://www.kaggle.com/abcsds/pokemon.

The Pokémon franchise encompasses video games, TV shows, movies, books, and a card game. This data set was drawn from the video game series and contains statistics about 721 Pokémon, or “pocket monsters.” In Pokémon games, the user plays as a trainer who collects, trades, and battles Pokémon to (a) collect all the Pokémon and (b) become the champion Pokémon trainer.

Each Pokémon has a primary type (some even have secondary types). Based on their type, a Pokémon is strong against some types, and vulnerable to others. (Think rock, paper, scissors.) A Fire-type Pokémon, for example, is vulnerable to Water-type Pokémon, but strong against Grass-type.

Figure 1: Fig 1. Vulpix, a Fire-type fox Pokémon from Generation 1.

The goal of this assignment is to build a statistical learning model that can predict the primary type of a Pokémon based on its generation, legendary status, and six battle statistics.

Read in the file and familiarize yourself with the variables using pokemon_codebook.txt.

library(tidymodels) library(tidyverse)

library(ISLR) # For the Smarket data set library(ISLR2) # For the Bikeshare data set library(klaR) # for naive bayes

library(discrim) library(poissonreg) library(corrr) library(forcats) library(corrplot) library(pROC) library(recipes) library(rsample) library(parsnip) library(workflows)

Exercise 1

Install and load the janitor package. Use its clean_names() function on the Pokémon data, and save the results to work with for the rest of the assignment. What happened to the data? Why do you think clean_names() is useful?

# install.packages(“janitor”) library(janitor) pokemon_raw &lt;- read.csv(“Pokemon.csv”) head(pokemon_raw)

## X. Name Type.1 Type.2 Total HP Attack Defense Sp..Atk

## 1 1 Bulbasaur Grass Poison 318 45 49 49 65

## 2 2 Ivysaur Grass Poison 405 60 62 63 80

## 3 3 Venusaur Grass Poison 525 80 82 83 100

## 4 3 VenusaurMega Venusaur Grass Poison 625 80 100 123 122

## 5 4 Charmander Fire 309 39 52 43 60

## 6 5 Charmeleon Fire

## Sp..Def Speed Generation Legendary

## 1 65 45 1 False

## 2 80 60 1 False

## 3 100 80 1 False

## 4 120 80 1 False

## 5 50 65 1 False

## 6 65 80 1 False 405 58 64 58 80

pokemon1 &lt;- clean_names(pokemon_raw) head(pokemon1)

## x name type_1 type_2 total hp attack defense sp_atk sp_def

## 1 1 Bulbasaur Grass Poison 318 45 49 49 65 65

## 2 2 Ivysaur Grass Poison 405 60 62 63 80 80

## 3 3 Venusaur Grass Poison 525 80 82 83 100 100

## 4 3 VenusaurMega Venusaur Grass Poison 625 80 100 123 122 120

## 5 4 Charmander Fire 309 39 52 43 60 50

## 6 5 Charmeleon Fire

## speed generation legendary

## 1 45 1 False

## 2 60 1 False

## 3 80 1 False

## 4 80 1 False

## 5 65 1 False

## 6 80 1 False 405 58 64 58 80 65

All column names are converted to lowercase and all of them are unique, also names that consist symbol dot are converted to underline.

It is useful because it makes names unique and consisting only of the characters (underline), numbers, and letters which is more efficiency and easier to read.

Exercise 2

Using the entire data set, create a bar chart of the outcome variable, type_1.

ggplot(pokemon1, aes(y= type_1)) +

geom_bar(stat = “count”) + ggtitle(“Bar Plot of Pokemon: type_1”) + xlab(“Type”)

Bar Plot of Pokemon: type_1

Water Steel

Rock

Psychic

Poison

Normal

Ice

Ground

Grass Ghost

Flying

Fire

Fighting

Fairy

Electric

Dragon

Dark

Bug

0 30 60 90

Type

How many classes of the outcome are there? Are there any Pokémon types with very few Pokémon? If so, which ones? There are 18 classes.

Flying type has very few Pokémon.

For this assignment, we’ll handle the rarer classes by simply filtering them out. Filter the entire data set to contain only Pokémon whose type_1 is Bug, Fire, Grass, Normal, Water, or Psychic.

After filtering, convert type_1 and legendary to factors.

pokemon &lt;- pokemon1[ which(pokemon1$type_1 == “Bug” | pokemon1$type_1 == “Fire” |

pokemon1$type_1 == “Grass” | pokemon1$type_1 == “Normal” | pokemon1$type_1 == “Water” | pokemon1$type_1 == “Psychic”), ]

pokemon &lt;- pokemon %&gt;% mutate(type_1 = factor(type_1), legendary = factor(legendary))

head(pokemon)

## x name type_1 type_2 total hp attack defense sp_atk sp_def

## 1 1 Bulbasaur Grass Poison 318 45 49 49 65 65

## 2 2 Ivysaur Grass Poison 405 60 62 63 80 80

## 3 3 Venusaur Grass Poison 525 80 82 83 100 100

## 4 3 VenusaurMega Venusaur Grass Poison 625 80 100 123 122 120

## 5 4 Charmander Fire 309 39 52 43 60 50

## 6 5 Charmeleon Fire

## speed generation legendary 405 58 64 58 80 65

## 1 45 1 False

## 2 60 1 False

## 3 80 1 False

## 4 80 1 False

## 5 65 1 False

## 6 80 1 False

Exercise 3

Perform an initial split of the data. Stratify by the outcome variable. You can choose a proportion to use. Verify that your training and test sets have the desired number of observations.

pokemon_split &lt;- pokemon %&gt;%

initial_split(strata = type_1, prop = 0.7)

pokemon_train &lt;- training(pokemon_split) pokemon_test &lt;- testing(pokemon_split) dim(pokemon_train)

## [1] 318 13

dim(pokemon_test)

## [1] 140 13

Next, use v-fold cross-validation on the training set. Use 5 folds. Stratify the folds by type_1 as well. Hint:

Look for a strata argument. Why might stratifying the folds be useful?

pokemon_fold &lt;- vfold_cv(pokemon_train, v = 5, strata = type_1)

Each re-sample will be created within the stratification variables where each fold is an appropriate representation of the original data.

Exercise 4

Set up a recipe to predict type_1 with legendary, generation, sp_atk, attack, speed, defense, hp, and sp_def.

• Dummy-code legendary and generation; • Center and scale all predictors.

pokemon_recipe &lt;- recipe(type_1 ~ legendary + generation + sp_atk + attack + speed + defense + hp + sp_def, pokemon_train) %&gt;%

step_dummy(legendary, generation) %&gt;% step_center(all_predictors()) %&gt;% step_scale(all_predictors())

Exercise 5

We’ll be fitting and tuning an elastic net, tuning penalty and mixture (use multinom_reg with the glmnet engine).

Set up this model and workflow. Create a regular grid for penalty and mixture with 10 levels each; mixture should range from 0 to 1. For this assignment, we’ll let penalty range from -5 to 5 (it’s log-scaled). How many total models will you be fitting when you fit these models to your folded data?

#cutting down levels to save runtimes

pokemon_spec &lt;- multinom_reg(mixture = tune(), penalty = tune()) %&gt;% set_mode(“classification”) %&gt;% set_engine(“glmnet”)

pokemon_workflow &lt;- workflow() %&gt;% add_recipe(pokemon_recipe) %&gt;% add_model(pokemon_spec)

penalty_grid &lt;- grid_regular(penalty(range = c(-5, 5)),

mixture(range = c(0, 1)), levels = 10)

penalty_grid

## # A tibble: 100 x 2

## penalty mixture

## &lt;dbl&gt; &lt;dbl&gt;

## 1 0.00001 0

## 2 0.000129 0

## 3 0.00167 0

## 4 0.0215 0

## 5 0.278 0

## 6 3.59 0

## 7 46.4 0

## 8 599. 0

## 9 7743. 0

## 10 100000 0

## # … with 90 more rows

There will be 500 (10 data in 10 levels within 5 folds: 5 x 10 x 10) models in total.

Exercise 6

Fit the models to your folded data using tune_grid().

Use autoplot() on the results. What do you notice? Do larger or smaller values of penalty and mixture produce better accuracy and ROC AUC?

tune_res &lt;- tune_grid(pokemon_workflow, resamples = pokemon_fold, grid = penalty_grid)

autoplot(tune_res)

Amount of Regularization

I notice that smaller value of penalty produces better accuracy and roc_auc, so does mixture.

Exercise 7

Use select_best() to choose the model that has the optimal roc_auc. Then use finalize_workflow(), fit(), and augment() to fit the model to the training set and evaluate its performance on the testing set.

best_model &lt;- select_best(tune_res, metric = “roc_auc”) pokemon_final &lt;- finalize_workflow(pokemon_workflow, best_model) pokemon_final_fit &lt;- fit(pokemon_final, data = pokemon_train)

predicted_data &lt;- augment(pokemon_final_fit, new_data = pokemon_test) %&gt;%

select(type_1,starts_with(“.pred”))

Exercise 8

Calculate the overall ROC AUC on the testing set.

predicted_data %&gt;% roc_auc(type_1, .pred_Bug:.pred_Water)

## # A tibble: 1 x 3

## .metric .estimator .estimate

## &lt;chr&gt; &lt;chr&gt; &lt;dbl&gt;

## 1 roc_auc hand_till 0.728

Then create plots of the different ROC curves, one per level of the outcome. Also make a heat map of the confusion matrix.

predicted_data %&gt;% roc_curve(type_1, .pred_Bug:.pred_Water) %&gt;%

autoplot()

augment(pokemon_final_fit, new_data = pokemon_test) %&gt;%

conf_mat(truth = type_1, estimate =.pred_class)%&gt;% autoplot(“heatmap”)

9 0 7 4 0 5

0

0 1

1 0 0 0 1

2 0 3 0

5 2 0 19 1 9

3 1 3 1 7

7 3

4 11 9 6 16

Bug

Fire

Grass

Normal

Psychic

Water

Bug Fire Grass Normal Psychic Water

Truth

#if all dark block are in diagonal it would be a good model.

What do you notice? How did your model do? Which Pokemon types is the model best at predicting, and which is it worst at? Do you have any ideas why this might be?
