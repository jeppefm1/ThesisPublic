#!/bin/bash
SESSION_NAME="matlab_session"

while true; do
     # Check if MATLAB is running
    if ! pgrep -f "matlab.*-nodisplay" > /dev/null; then
        echo "MATLAB crashed. Creating new session..."
        tmux kill-session -t $SESSION_NAME
        tmux new-session -d -s $SESSION_NAME

        # Start MATLAB in the correct directory and run the script
        tmux send-keys -t $SESSION_NAME "cd jeppes_project/Thesis/rigshospitalet/MATLAB && jeppes_project/matlab2024b/bin/matlab -nodisplay -nosplash -nodesktop -r \"try; reconstructScans(); catch; end; exit;\"" C-m
    fi


    sleep 100  # Avoid CPU overuse
done
