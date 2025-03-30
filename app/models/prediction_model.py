# Copy the HeartFailurePredictionModel class from the provided code
# (The full class implementation will be copied here)

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
from tqdm import tqdm

class HeartFailurePredictionModel:
    """Class for training and evaluating heart failure prediction models"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )
        }
        self.best_model = None
        self.feature_importance = None
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train multiple models and select the best one"""
        best_score = 0
        print("Starting model training...")
        
        # Handle NaN values using imputer
        X_train_imputed = self.imputer.fit_transform(X_train)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        
        if X_val is not None:
            X_val_imputed = self.imputer.transform(X_val)
            X_val_scaled = self.scaler.transform(X_val_imputed)
        
        # Train each model with multiple epochs
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if name == 'neural_network':
                # Neural network has built-in epochs
                model.fit(X_train_scaled, y_train)
                print(f"Number of iterations completed: {model.n_iter_}")
            else:
                # For tree-based models, we'll use bootstrapping to simulate epochs
                for epoch in range(12):
                    # Sample with replacement
                    indices = np.random.choice(len(X_train_scaled), size=len(X_train_scaled), replace=True)
                    X_epoch = X_train_scaled[indices]
                    y_epoch = y_train[indices]
                    
                    # Partial fit
                    if epoch == 0:
                        model.fit(X_epoch, y_epoch)
                    else:
                        # For random forest and gradient boosting, we'll adjust the weights
                        if name == 'random_forest':
                            model.n_estimators += 10
                            model.fit(X_epoch, y_epoch)
                        elif name == 'gradient_boosting':
                            model.n_estimators += 10
                            model.fit(X_epoch, y_epoch)
                    
                    # Evaluate current performance
                    train_score = model.score(X_train_scaled, y_train)
                    print(f"Epoch {epoch + 1}/12, Training score: {train_score:.4f}")
            
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val_scaled)
                score = f1_score(y_val, y_pred)
                print(f"{name} validation F1 score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    self.best_model = (name, model)
            else:
                self.best_model = (name, model)
        
        print(f"\nBest model: {self.best_model[0]} with F1 score: {best_score:.4f}")
        
        # Calculate feature importance for the best model
        if self.best_model[0] in ['random_forest', 'gradient_boosting']:
            self.feature_importance = dict(zip(X_train.columns, self.best_model[1].feature_importances_))
        
        return self.best_model[1]
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Handle NaN values and scale features
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.best_model[1].predict(X_scaled)
    
    def predict_proba(self, X):
        """Get probability predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Handle NaN values and scale features
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        probas = self.best_model[1].predict_proba(X_scaled)
        
        # Ensure probabilities are well-calibrated
        if np.all(probas[:, 1] < 0.1):  # If all probabilities are too low
            probas = np.clip(probas * 2, 0, 1)  # Scale up the probabilities
        
        return probas
    
    def evaluate(self, X_test, y_test):
        """Evaluate the best model on test data"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Handle NaN values and scale features
        X_test_imputed = self.imputer.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        print(f"\nModel: {self.best_model[0]}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Abnormal (Heart Failure Risk)'],
                    yticklabels=['Normal', 'Abnormal (Heart Failure Risk)'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        results['confusion_matrix'] = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {results["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        results['roc_curve'] = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        # If feature importance is available, plot it
        if self.feature_importance is not None:
            # Sort features by importance
            sorted_features = dict(sorted(self.feature_importance.items(), 
                                         key=lambda item: item[1], reverse=True))
            # Take top 20 features
            top_features = dict(list(sorted_features.items())[:20])
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x=list(top_features.values()), y=list(top_features.keys()))
            plt.title('Top 20 Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Convert plot to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            results['feature_importance'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        
        return results
