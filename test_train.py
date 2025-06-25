import subprocess
import os

# Change to the complaints-classifications directory
os.chdir('/home/legion/Documents/Legion/Development/pydjango/ai-practice/complaints-classifications')

# Run the train_classifier.py script
print("Running train_classifier.py...")
result = subprocess.run(['python', 'train_classifier.py'], 
                        capture_output=True, 
                        text=True)

# Print the output
print("STDOUT:")
print(result.stdout)

print("STDERR:")
print(result.stderr)

print("Return code:", result.returncode)