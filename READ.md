# Default live animation (updates every 500ms)
python main.py --live-animation

# Faster animation (200ms updates)
python main.py --live-animation --animation-speed 200

# Slower animation for detailed observation (1000ms)
python main.py --live-animation --animation-speed 1000

# Simple maze with fast animation
python main.py --maze simple --episodes 200 --live-animation --animation-speed 300

python main.py --live-animation --animation-speed 100 --training-interval 1000 --episodes 1000

# Watch training with visualization every 10 episodes
python main.py --watch-training

# Watch every 5 episodes
python main.py --watch-training --training-interval 5

# Watch training on simple maze
python main.py --maze simple --episodes 200 --watch-training --training-interval 20

# Train and show interactive journey
python main.py --show-journey

# Just show journey of a saved model
python main.py --load-qtable my_model.npy --episodes 0 --show-journey

# Train on simple maze and show journey
python main.py --maze simple --episodes 500 --show-journey

# Show detailed Q-table analysis with visualizations
python main.py --analyze-qtable

# Analysis without regular visualization
python main.py --analyze-qtable --no-viz