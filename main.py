import joblib
from data_integration import integrate_data, save_integrated_data, load_integrated_data
from modeling import train_and_evaluate_models
import plotly.graph_objects as go

def main():
    # Step 1: Integrate data
    print("Integrating data...")
    integrated_data = integrate_data()
    save_integrated_data(integrated_data)

    # Step 2: Train and evaluate models
    print("Training and evaluating models...")
    model_results = train_and_evaluate_models(integrated_data)

    # Step 3: Print model performance
    print("Model performance:")
    for model_name, metrics in model_results.items():
        print(f"\\n{model_name.capitalize()} Model:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

    # Step 4: Load the best model
    best_model = joblib.load('best_model.joblib')

    # Step 5: Make predictions on new data
    print("\\nMaking predictions on new data...")
    new_data = load_integrated_data()  # In a real scenario, this would be fresh data
    X_new = new_data.drop(["result", "timestamp", "team_name", "opponent_name"], axis=1)
    predictions = best_model.predict(X_new)

    # Step 6: Visualize predictions
    visualize_predictions(new_data, predictions)

def visualize_predictions(data, predictions):
    data['predicted_result'] = predictions

    # Create a scatter plot of actual vs predicted results
    fig = go.Figure()

    for result in ['W', 'D', 'L']:
        mask = data['result'] == result
        fig.add_trace(go.Scatter(
            x=data[mask]['timestamp'],
            y=data[mask]['predicted_result'],
            mode='markers',
            name=f'Actual {result}',
            marker=dict(size=10, opacity=0.6)
        ))

    fig.update_layout(
        title='Actual vs Predicted Match Results',
        xaxis_title='Date',
        yaxis_title='Predicted Result',
        legend_title='Actual Result'
    )

    fig.write_html("predictions_visualization.html")
    print("Predictions visualization saved to predictions_visualization.html")

if __name__ == "__main__":
    main()
