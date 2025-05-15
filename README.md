# Card-fraud-detection
## Problem discussion
Currently in Vietnam, people use banking services and bank cards for transactions, QR code scanning, and more. With thousands or even hundreds of transactions taking place every hour, it led us to wonder how often invalid or fraudulent transactions occur — especially with credit cards. From that, we decided to predict fraudulent card transactions with the help of machine learning.

## Understading and Defining fraud

Credit card fraud is the most common form of identity theft. With an estimated 1.5 billion credit cards in circulation in the United States alone, it's no surprise that millions fall victim each year.

Some criminals commit fraud using lost or stolen credit cards. Others carry out illegal transactions without ever needing to have the physical card in hand. This type of fraud—called card-not-present fraud—only requires the criminal to obtain basic account details to access the victim’s funds.

## How Credit Card Fraud Works?

Any business, regardless of size, can be vulnerable to credit card theft and fraud. Here are several common methods:

- Lost or Stolen Cards
Criminals may obtain credit cards by finding them after they’ve been lost or by stealing them directly from individuals.
While they may not always be able to use the card in person at point-of-sale terminals that require a PIN, they can still use the card details to make online purchases.

- Card-Not-Present Fraud
This form of fraud doesn’t require possession of the physical card. Instead, the fraudster obtains basic information such as the cardholder’s name, card number, and expiration date. With these details, they can commit fraud via mail, phone, or online transactions.

- Counterfeit, Cloned, or Tampered Cards
Devices known as skimmers can illegally capture data from a card’s magnetic stripe. Criminals can then encode this information onto fake, cloned, or altered cards.
It can be difficult to detect the difference between a normal card reader or ATM and one that has been tampered with using a skimmer. (As a precaution, always check if a card reader feels loose or looks unusual before using it.)

- Application Fraud
Instead of using an existing card, fraudsters may apply for new credit in someone else’s name. They do this using the victim’s personal details—such as full name, date of birth, address, and Social Security number—and may even forge supporting documents to verify the application.

- Account Takeover
Once criminals have enough personal information, they may contact the credit card issuer pretending to be the account holder. By providing details such as previous purchases, passwords, and card information, they may convince the company to change the account address or report the card as lost or stolen—prompting the issuer to mail a new card to the updated (fraudulent) address.
This method is often referred to as social engineering.

- Interception of Mailed Cards
If a credit card company sends a new or replacement card through the mail, fraudsters may intercept or steal it from the recipient’s mailbox.
To prevent this, most issuers use plain envelopes with no branding when mailing cards.


## data dictionary
The dataset can be download using this [link](https://www.kaggle.com/mlg-ulb/creditcardfraud)

The data set includes credit card transactions made by European cardholders over a period of two days in September 2013. Out of a total of 2,84,807 transactions, 492 were fraudulent. This data set is highly unbalanced, with the positive class (frauds) accounting for 0.172% of the total transactions. The data set has also been modified with Principal Component Analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The feature 'class' represents class labelling, and it takes the value 1 in cases of fraud and 0 in others.

## Solution approach
- step 1 : understanding data
- step 2 : data cleaning (handling missing value, outlier treatment)
- step 3 : exploratory data analysis (Univariate analysis , Bivariate analysis)
- step 4 : Check the skewness of the data and mitigate it for fair analysis ,
Handling data imbalance as we see only 0.172% records are the fraud
- step 5 : Split the data into train and test set (scale the data)
- step 6 : train model(logistic regression, decision tree, etc)
- step 7 : model evaluation (ROC-AUC score ,  F1, recall,...)

