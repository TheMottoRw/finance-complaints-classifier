from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
import os
import joblib
import numpy as np
from predict_classifier import predict_complaint
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/complaints_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Department model
class Department(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    user_type = db.Column(db.String(20), nullable=False, default='user')  # 'admin', 'user', etc.
    department_id = db.Column(db.Integer, db.ForeignKey('department.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    department = db.relationship('Department', backref=db.backref('users', lazy=True))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Complaint model
class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), nullable=True)
    department_id = db.Column(db.Integer, db.ForeignKey('department.id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(50), nullable=False, default='pending')  # pending, in-progress, resolved, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('complaints', lazy=True))
    department = db.relationship('Department', backref=db.backref('complaints', lazy=True))

# ComplaintHandling model
class ComplaintHandling(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), nullable=False, default='unread')  # unread, read, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    sender = db.relationship('User', foreign_keys=[sender_id], backref=db.backref('sent_messages', lazy=True))
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref=db.backref('received_messages', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already exists')
            return redirect(url_for('signup'))

        # Create new user
        new_user = User(
            name=name,
            email=email,
            user_type='customer',
            department_id=None
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        # Log in the new user
        login_user(new_user)
        return redirect(url_for('dashboard'))

    # Get all departments for the form
    departments = Department.query.all()
    return render_template('signup.html', departments=departments)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # For regular users, show only their complaints
    if current_user.user_type == 'user':
        complaints = Complaint.query.filter_by(user_id=current_user.id).all()
    # For admin users, show all complaints
    else:
        complaints = Complaint.query.all()

    return render_template('dashboard.html', complaints=complaints, user_type=current_user.user_type)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        department_id = request.form.get('department_id')

        # Predict category using the model
        category = predict_complaint(content)
        department_id = Department.query.filter_by(name=category).first().id if category else None

        # Save complaint to database
        new_complaint = Complaint(
            title=title,
            content=content,
            category=category,
            department_id=department_id if department_id else None,
            user_id=current_user.id,
            status='pending'
        )
        db.session.add(new_complaint)
        db.session.commit()

        return render_template('predict.html', prediction=category, complaint_title=title, complaint_content=content)

    # Get all departments for the form
    departments = Department.query.all()
    return render_template('predict.html', departments=departments)


@app.route('/edit_complaint_category', methods=['GET','POST'])
def edit_complaint_category():
    """
    Route to handle updating complaint categories
    Expects POST data: complaint_id, new_category, reason (optional)
    """
    try:
        if request.method != 'POST':
            complaints = Complaint.query.all()  # or filter by current user if needed
            return render_template("edit_complaint_category.html",complaints=complaints)
        # Get form data
        complaint_id = request.form.get('complaint_id')
        new_category = request.form.get('new_category')
        reason = request.form.get('reason', '')  # Optional field

        # Validate required fields
        if not complaint_id or not new_category:
            flash('Missing required information. Please try again.', 'error')
            return redirect(url_for('manage_complaints'))

        # Find the complaint in the database
        complaint = Complaint.query.get(complaint_id)

        if not complaint:
            flash('Complaint not found.', 'error')
            return redirect(url_for('manage_complaints'))

        # Store old category for logging/feedback
        old_category = complaint.category

        # Update the complaint category
        complaint.category = new_category

        # Optional: If you want to store the reason for the change
        # You might need to add a 'change_reason' or 'notes' field to your model
        if hasattr(complaint, 'change_reason'):
            complaint.change_reason = reason

        # Optional: Add timestamp for when category was changed
        if hasattr(complaint, 'category_updated_at'):
            from datetime import datetime
            complaint.category_updated_at = datetime.utcnow()

        # Save changes to database
        db.session.commit()

        # Success message
        flash(f'Complaint category successfully updated from "{old_category}" to "{new_category}".', 'success')

    except Exception as e:
        # Handle any database errors
        db.session.rollback()
        flash('An error occurred while updating the complaint category. Please try again.', 'error')
        print(f"Error updating complaint category: {str(e)}")  # For debugging

    # Redirect back to the manage complaints page
    return redirect(url_for('manage_complaints'))


@app.route('/manage_complaints')
def manage_complaints():
    """
    Route to display the manage complaints page
    """
    # Get all complaints from database
    # Adjust this query based on your user authentication system
    complaints = Complaint.query.all()  # or filter by current user if needed

    return render_template('manage_complaints.html', complaints=complaints)
def create_default_admin():
    """Create a default admin user if it doesn't exist"""
    admin = User.query.filter_by(email='asua@yopmail.com').first()
    if not admin:

        # Create admin user
        admin = User(
            name='Admin',
            email='asua@yopmail.com',
            user_type='admin',
            department_id=None
        )
        admin.set_password('12345')
        db.session.add(admin)
        db.session.commit()
        print("Default admin user created.")

if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
        create_default_admin()
    app.run(debug=True)
