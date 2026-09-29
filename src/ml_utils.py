import functools
from sklearnex import config_context
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
#Global Variable of gpu acceleration
USE_GPU = False

def set_gpu_enabled(enabled: bool):

    global USE_GPU
    USE_GPU = enabled
    if USE_GPU:
        print("GPU acceleration enabled")
    
    
def create_pipeline(preprocessor, classifier, feature_selector=None):
    """Builds a pipeline conditionally handling optional feature selection."""
    steps = [('preprocessing', preprocessor)]
    if feature_selector is not None:
        steps.append(('feature_selection', feature_selector))
    steps.append(('classifier', classifier))
    return Pipeline(steps)

def gpu_acceleration(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        if USE_GPU:
            try:
                with config_context(target_offload='gpu:0'):
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                print(f"GPU execution failed ({e}). Falling back to CPU...")        
        result = func(*args, **kwargs)
        return result
    return inner


@gpu_acceleration
def train_model(search_object, x_train, y_train):
    """Fits the search object using training data and returns it."""
    search_object.fit(x_train, y_train)
    return search_object



def evaluate_results(gs, x_test, y_test, pos_label=1):
    best_model = gs.best_estimator_
    classifier = best_model.named_steps.get('classifier', best_model)
    model_name = classifier.__class__.__name__

    print(f"\n=== {model_name} | Best CV Score: {gs.best_score_:.4f} ===")

    params = [f"{k}={v}" for k, v in classifier.get_params().items() if v is not None and v != 'warn']
    print(f"Params: {', '.join(params)}")

    predictions = best_model.predict(x_test)
    probabilities = best_model.predict_proba(x_test)[:, 1]

    test_acc = best_model.score(x_test, y_test)
    precision = precision_score(y_test, predictions, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_test, predictions, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label=pos_label, zero_division=0)
    roc_auc = roc_auc_score((y_test == pos_label).astype(int), probabilities)
    cm = confusion_matrix(y_test, predictions)

    print(f"Metrics  -> Acc: {test_acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")
    print(f"Conf Mat -> [[{cm[0, 0]}, {cm[0, 1]}], [{cm[1, 0]}, {cm[1, 1]}]]")