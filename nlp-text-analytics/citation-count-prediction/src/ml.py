####################################
# Author: S. A. Owerre
# Date modified: 10/05/2021
# Class: Supervised Regression ML
####################################

import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

class RegressionModels:
    """This class is used for training supervised regression models."""

    def __init__(self):
        """Parameter initialization."""

    def eval_metric_cv(
        self, 
        model, 
        X_train, 
        y_train, 
        cv_fold, 
        model_nm=None
    ):
        """Cross-validation on the training set.

        Parameters
        ----------
        model: supervised classification model
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold

        Returns
        -------
        Performance metrics on the cross-validation training set
        """

        # fit the training set
        model.fit(X_train, y_train)

        # make prediction on k-fold cross validation set
        y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv_fold)

        # print results
        print('{}-Fold cross-validation results for {}'.format(
            str(cv_fold), 
            str(model_nm)
            )
        )
        print('-' * 45)
        print(self.error_metrics(y_train, y_pred_cv))
        print('-' * 45)
    
    def plot_mae_rsme_svr(self, X_train, y_train, cv_fold):
        """Plot of cross-validation MAE and RMSE for SVR.

        Parameters
        ----------
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold

        Returns
        -------
        matplolib figure of MAE & RMSE
        """
        C_list = [2**x for x in range(-2,11,2)]
        gamma_list = [2**x for x in range(-7,-1,2)]
        mae_list = [
            pd.Series(0.0, 
            index=range(len(C_list))
            ) for _ in range(len(gamma_list)
            )
        ]
        rmse_list = [
            pd.Series(0.0,
            index=range(len(C_list))
            ) for _ in range(len(gamma_list)
            )
        ]
        axes_labels = ['2^-2', '2^0', '2^2', '2^4', '2^6', '2^8', '2^10']
        gamma_labels = ['2^-7', '2^-5', '2^-3']
        plt.rcParams.update({'font.size': 15})
        _, (ax1, ax2) = plt.subplots(1,2, figsize = (18,6))

        for i, val1 in enumerate(gamma_list):
            for j, val2 in enumerate(C_list):
                model = SVR(C = val2, gamma = val1, kernel = 'rbf')
                model.fit(X_train, y_train)
                y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv_fold)
                mae_list[i][j] = self.mae(y_train, y_pred_cv)
                rmse_list[i][j] = self.rmse(y_train, y_pred_cv)
            mae_list[i].plot(
                label="gamma="+str(gamma_labels[i]), 
                marker = "o", 
                linestyle="-", ax=ax1
            )
            rmse_list[i].plot(
                label="gamma="+str(gamma_labels[i]), 
                marker="o", 
                linestyle="-", 
                ax=ax2
            )

        ax1.set_xlabel("C", fontsize = 15)
        ax1.set_ylabel("MAE", fontsize = 15)
        ax1.set_title("{}-Fold Cross-Validation with RBF kernel SVR".format(
            cv_fold
            ), 
            fontsize=15
        )
        ax1.set_xticklabels(axes_labels)
        ax1.set_xticks(range(len(C_list)))
        ax1.legend(loc = 'best')
        ax2.set_xlabel("C", fontsize = 15)
        ax2.set_ylabel("RSME", fontsize = 15)
        ax2.set_title("{}-Fold Cross-Validation with RBF kernel SVR".format(
            cv_fold
            ), 
            fontsize=15
        )
        ax2.set_xticks(range(len(C_list)))
        ax2.set_xticklabels(axes_labels)
        ax2.legend(loc = 'best')
        plt.show()
        
    def eval_metric_test(self, y_pred, y_true, model_nm = None):
        """Predictions on the test set.

        Parameters
        ----------
        y_pred: training set class labels
        y_true: test set class labels

        Returns
        -------
        Performance metrics on the test set
        """
        # Print results
        print('Test prediction results for {}'.format(model_nm))
        print('-' * 45)
        print(self.error_metrics(y_true, y_pred))
        print('-' * 45)
        
    def diagnostic_plot(self, y_pred, y_true, ylim = None):
        """Diagnostic plot
        
        Parameters
        ----------
        y_pred: predicted labels
        y_true: true labels

        Returns
        -------
        Matplolib figure
        """
        # compute residual and metrics
        residual = (y_true - y_pred)
        r2 = np.round(self.r_squared(y_true, y_pred), 3)
        rm = np.round(self.rmse(y_true, y_pred), 3)
        
        # plot figures
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
        ax1.scatter(y_pred, residual, color ='b')
        ax1.set_xlim([-0.1, 14])
        ax1.set_ylim(ylim)
        ax1.hlines(y=0, xmin=-0.1, xmax=14, lw=2, color='k')
        ax1.set_xlabel('Predicted values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs. Predicted values')
        ax2.scatter(y_pred, y_true, color='b')
        ax2.plot([-0.3, 9], [-0.3, 9], color='k')
        ax2.set_xlim([-0.3, 9])
        ax2.set_ylim([-0.3, 9])
        ax2.text(
            2,
            7,
            r'$R^2 = {},~ RMSE = {}$'.format(str(r2), str(rm)),
            fontsize=20,
        )
        ax2.set_xlabel('Predicted values')
        ax2.set_ylabel('True values')
        ax2.set_title('True values vs. Predicted values')
    
    def error_metrics(self, y_true, y_pred):
        """Print out error metrics."""
        r2 = self.r_squared(y_true, y_pred)
        mae = self.mae(y_true, y_pred)
        rmse = self.rmse(y_true, y_pred)
        result = {
            'MAE = {}'.format(np.round(mae,3)),
            'RMSE = {}'.format(np.round(rmse,3)),
            'R^2 = {}'.format(np.round(r2,3)),
        }
        return result

    def mae(self, y_test, y_pred):
        """Mean absolute error.
        
        Parameters
        ----------
        y_test: test set label
        y_pred: prediction label

        Returns
        -------
        Mean absolute error
        """
        mae = np.mean(np.abs((y_test - y_pred)))
        return mae

    def rmse(self, y_test, y_pred):
        """Root mean squared error.
        
        Parameters
        ----------
        y_test: test set label
        y_pred: prediction label

        Returns
        -------
        Root mean squared error
        """
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        return rmse

    def r_squared(self, y_test, y_pred):
        """r-squared (coefficient of determination).
        
        Parameters
        ----------
        y_test: test set label
        y_pred: prediction label

        Returns
        -------
        r-squared
        """
        mse = np.mean((y_test - y_pred)**2)  # mean squared error
        var = np.mean((y_test - np.mean(y_test))**2)  # sample variance
        r_squared = 1 - mse / var
        return r_squared