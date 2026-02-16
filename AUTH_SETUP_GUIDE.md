# 🚀 Authentication Setup Guide

## ✅ What's Been Implemented

Your backend now has production-ready authentication with:
- User registration and login
- JWT token-based authentication
- Database persistence for all chat messages
- Multi-user support with conversation isolation
- Secure password hashing with bcrypt

## 📋 Setup Steps

### 1. Create `.env` File

Copy the example file and update with your credentials:

```bash
cd backend
copy .env.example .env
```

**Important:** Update these values in your `.env`:
- `JWT_SECRET_KEY` - Generate a secure random key (run the command below)
- `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` - Your PostgreSQL credentials
- `GROQ_API_KEY` - Your Groq API key
- `OPENAI_API_KEY` - Your OpenAI API key

**Generate a secure JWT secret:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Create Database Tables

1. Open **PgAdmin**
2. Connect to your PostgreSQL database
3. Open the query tool
4. Open `backend/database_setup.sql`
5. Execute the entire script
6. Verify all 5 tables were created:
   - `users`
   - `conversations`
   - `messages`
   - `message_context_items`
   - `refresh_tokens`

### 3. Install Dependencies (if not already done)

```bash
cd backend
pip install python-jose[cryptography] passlib[bcrypt] email-validator
```

### 4. Start the Backend

```bash
cd backend
python run.py
```

The API will be available at: `http://localhost:8000`
API docs at: `http://localhost:8000/api/docs`

## 🔐 New API Endpoints

### Authentication Endpoints

#### Register New User
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecurePass123!",
  "full_name": "John Doe"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGc...",
  "refresh_token": "eyJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "email": "user@example.com",
    "username": "johndoe",
    "full_name": "John Doe",
    "created_at": "2026-02-16T10:30:00",
    "is_active": true,
    "is_verified": false
  }
}
```

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Response:** Same as registration

#### Refresh Access Token
```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### Get User Profile
```http
GET /auth/me
Authorization: Bearer {access_token}
```

#### List User's Conversations
```http
GET /auth/conversations
Authorization: Bearer {access_token}
```

#### Logout
```http
POST /auth/logout
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "refresh_token": "eyJhbGc..."
}
```

### Protected Chat Endpoints (Require Authentication)

#### Send Message
```http
POST /api/chat
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "message": "What is the payment process?",
  "include_context": true,
  "top_k": 8,
  "conversation_id": "optional-uuid-or-null-for-new-chat"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Based on the knowledge base...",
  "context": "...",
  "context_items": [...],
  "context_count": 3,
  "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
  "timestamp": "2026-02-16T10:30:00"
}
```

#### Get Conversation History
```http
GET /api/history?conversation_id={uuid}&limit=50&offset=0
Authorization: Bearer {access_token}
```

#### Edit Message
```http
POST /api/edit-message
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "message": "Updated question text",
  "message_index": 0,
  "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
  "include_context": true,
  "top_k": 8
}
```

#### Delete Conversation
```http
DELETE /api/conversation/{conversation_id}
Authorization: Bearer {access_token}
```

## 🧪 Testing with cURL

### 1. Register a user
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "Test123!",
    "full_name": "Test User"
  }'
```

### 2. Login (save the access_token)
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "Test123!"
  }'
```

### 3. Send a chat message (use your access_token)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN_HERE" \
  -d '{
    "message": "What is the payment process?",
    "include_context": true
  }'
```

## 🔍 Troubleshooting

### "Database connection failed"
- Check your `.env` file has correct database credentials
- Ensure PostgreSQL is running
- Verify you can connect with PgAdmin

### "Failed to create conversation"
- Make sure you ran the database setup SQL script
- Check all tables exist in your database
- Verify the `users` table has your user record

### "Invalid or expired token"
- Your access token expires after 30 minutes
- Use the refresh token endpoint to get a new access token
- Make sure JWT_SECRET_KEY is set in `.env`

### "User already exists"
- Email or username is already taken
- Use a different email/username or login instead

## 📊 Database Schema

```
users
├── id (UUID, PK)
├── email (VARCHAR, UNIQUE)
├── username (VARCHAR, UNIQUE)
├── password_hash (VARCHAR)
├── full_name (VARCHAR)
├── created_at (TIMESTAMP)
├── last_login (TIMESTAMP)
├── is_active (BOOLEAN)
└── is_verified (BOOLEAN)

conversations
├── id (UUID, PK)
├── user_id (UUID, FK -> users.id)
├── title (VARCHAR)
├── created_at (TIMESTAMP)
├── updated_at (TIMESTAMP)
└── is_archived (BOOLEAN)

messages
├── id (UUID, PK)
├── conversation_id (UUID, FK -> conversations.id)
├── role (VARCHAR: 'user'|'assistant'|'system')
├── content (TEXT)
├── context_used (TEXT)
├── context_count (INTEGER)
├── created_at (TIMESTAMP)
├── edited_at (TIMESTAMP)
└── is_edited (BOOLEAN)

message_context_items
├── id (UUID, PK)
├── message_id (UUID, FK -> messages.id)
├── node_id (UUID, FK -> knowledge_node.id)
├── similarity_score (FLOAT)
└── position (INTEGER)

refresh_tokens
├── id (UUID, PK)
├── user_id (UUID, FK -> users.id)
├── token_hash (VARCHAR)
├── expires_at (TIMESTAMP)
├── created_at (TIMESTAMP)
└── revoked (BOOLEAN)
```

## 🎯 Next Steps

1. **Test the backend** - Follow the cURL examples above
2. **Update the frontend** - Implement login/register UI and token management
3. **Azure deployment** - Update your Azure app settings with the new environment variables

## 💡 Key Changes from Before

| Before | After |
|--------|-------|
| ❌ In-memory storage (lost on restart) | ✅ PostgreSQL database persistence |
| ❌ No authentication | ✅ JWT-based authentication |
| ❌ Shared conversations | ✅ User-isolated conversations |
| ❌ Azure instance issues | ✅ Production-ready, multi-instance safe |

---

**Need help?** Check the FastAPI interactive docs at `http://localhost:8000/api/docs`
