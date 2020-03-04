from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score,recall_score, f1_score, roc_curve, auc, fbeta_score, make_scorer, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Use class to cleanly store model predictions and output metrics, and create visualiztions. 
class predict_and_get_metrics():
    def __init__(self, model_name, model, X_train_ftrs, X_val_ftrs, y_train_real, y_val_real):
        self.model_name = model_name
        self.y_train_real = y_train_real
        self.y_val_real = y_val_real
        
        # Predict output values for train and validation set. Store predicted probabilites for use in AUROC graphs.
        self.y_train_pred = model.predict(X_train_ftrs)
        y_train_pred_prob = model.predict_proba(X_train_ftrs)
        self.y_train_pred_prob = y_train_pred_prob[:,1]
        
        self.y_val_pred = model.predict(X_val_ftrs)
        y_val_pred_prob = model.predict_proba(X_val_ftrs)
        self.y_val_pred_prob = y_val_pred_prob[:,1]
        
        # Get and store key metrics for model performance on train and validation set.
        self.train_auroc = roc_auc_score(y_train_real, self.y_train_pred_prob)
        self.val_auroc = roc_auc_score(y_val_real, self.y_val_pred_prob)
        
        self.train_precision = precision_score(y_train_real, self.y_train_pred)
        self.val_precision = precision_score(y_val_real, self.y_val_pred)
        
        self.train_recall = recall_score(y_train_real, self.y_train_pred)
        self.val_recall = recall_score(y_val_real, self.y_val_pred)

        self.train_fbeta = fbeta_score(y_train_real, self.y_train_pred, beta=0.8)
        self.val_fbeta = fbeta_score(y_val_real, self.y_val_pred, beta=0.8)

        self.train_accuracy = accuracy_score(y_train_real, self.y_train_pred)
        self.val_accuracy = accuracy_score(y_val_real, self.y_val_pred)
        
        # Create metrics dictionary with metrics for validation and training set defined above.
        self.metrics = ['Accuracy', 'AUROC', 'Precision', 'Recall', 'F beta']
        self.train_outputs = [self.train_accuracy, self.train_auroc, self.train_precision, 
                              self.train_recall, self.train_fbeta]
        self.val_outputs = [self.val_accuracy, self.val_auroc, self.val_precision, 
                            self.val_recall, self.val_fbeta]
        metrics_dict = {'Metric': self.metrics, 'Train Set': self.train_outputs, 
                        'Validation Set': self.val_outputs}
        
        # Output these metrics in a clean pandas dataframe.
        self.metrics_df = pd.DataFrame(metrics_dict).set_index('Metric')
        
    #Add functionality to quickly plot AUROC curves for each model.    
    def plot_auroc(self):
        def draw_curve(y_real,y_pred_prob, name, color):
            fpr, tpr, threshold = roc_curve(y_real, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            line_out = plt.plot(fpr, tpr, color, label = f'{name} AUC = {roc_auc}')
            return line_out
        # Draw curves for both training and validation set performance.
        draw_curve(self.y_train_real, self.y_train_pred_prob, 'Train', 'r')
        draw_curve(self.y_val_real, self.y_val_pred_prob, 'Val', 'b')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    
    def plot_precision_recall(self):    
        # Create function to easily draw precision recall curve
        def draw_curve(y_real, y_pred_prob,  y_pred, line_color, dataset):
            precision, recall, thresh = precision_recall_curve(y_real, y_pred_prob)
            f1 = f1_score(y_real, y_pred)
            auc_val = auc(recall, precision)
            # Draw precision recall curve
            plt.plot(recall, precision, linewidth='1', label=self.model_name+' '+dataset, color=line_color)
        draw_curve(self.y_train_real, self.y_train_pred_prob,  self.y_train_pred, line_color='blue', dataset='train')
        draw_curve(self.y_val_real, self.y_val_pred_prob,  self.y_val_pred, line_color='red', dataset='val')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

    # Produce precision/recall and AUROC plots and print dataframe
    def show_out(self):
        self.plot_auroc(), self.plot_precision_recall()
        #returning rather than printing dataframe allows for clean printing (though this is unconventional code)
        return self.metrics_df

# Create function to select highest accuracy and fbeta scores from lists of metric output objects.
def get_winner(master_list):
    highest_accuracy = 0
    highest_fbeta = 0
    for output_object in master_list:
        if output_object.val_accuracy > highest_accuracy:
            highest_accuracy = output_object.val_accuracy
            accuracy_winner = output_object
        if output_object.val_fbeta > highest_fbeta:
            highest_fbeta = output_object.val_fbeta
            fbeta_winner = output_object
    # If same model wins both parameters, print model name, parameter values, and dataframe.
    if fbeta_winner == accuracy_winner:
        print(f"Highest accuracy ({highest_accuracy}) and highest fbeta score ({highest_fbeta}) achieved by {fbeta_winner.model_name}:")
        # Using return on dataframe allows for crisp display; Must adjust if not using notebook. 
        return(fbeta_winner.metrics_df)
    # If different model wins each parameter, print model name and dataframe of each for comparison.
    else: 
        print(f"Highest accuracy ({highest_accuracy}) achieved by {accuracy_winner.model_name}:")
        print(accuracy_winner.metrics_df)
        print(f"Highest fbeta score ({highest_fbeta}) achieved by {fbeta_winner.model_name}:")
        print(fbeta_winner.metrics_df)
