#!/usr/bin/env python3
"""
🧪 TELEGRAM CONNECTION TEST
Test if your BOT_TOKEN and CHAT_ID work
"""

import os
import requests

BOT_TOKEN = os.environ.get('BOT_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

print("="*60)
print("🧪 TELEGRAM CONNECTION TEST")
print("="*60)

# Check if secrets exist
print(f"\n1️⃣ Checking secrets...")
if not BOT_TOKEN:
    print("   ❌ BOT_TOKEN is missing or empty!")
    print("   → Go to: Settings → Secrets → Actions")
    print("   → Add secret: BOT_TOKEN")
else:
    print(f"   ✅ BOT_TOKEN exists (length: {len(BOT_TOKEN)} chars)")

if not CHAT_ID:
    print("   ❌ CHAT_ID is missing or empty!")
    print("   → Go to: Settings → Secrets → Actions")
    print("   → Add secret: CHAT_ID")
else:
    print(f"   ✅ CHAT_ID exists (value: {CHAT_ID})")

if not BOT_TOKEN or not CHAT_ID:
    print("\n❌ Cannot proceed without secrets!")
    exit(1)

# Test bot token validity
print(f"\n2️⃣ Testing bot token...")
try:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    response = requests.get(url, timeout=10)
    data = response.json()
    
    if data.get('ok'):
        bot_info = data.get('result', {})
        print(f"   ✅ Bot token is VALID!")
        print(f"   Bot name: {bot_info.get('first_name')}")
        print(f"   Bot username: @{bot_info.get('username')}")
    else:
        print(f"   ❌ Bot token is INVALID!")
        print(f"   Error: {data}")
except Exception as e:
    print(f"   ❌ Error connecting to Telegram: {e}")
    exit(1)

# Test sending message
print(f"\n3️⃣ Testing message sending...")
try:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "🧪 **TEST MESSAGE**\n\nIf you see this, your Telegram integration is working!\n\n✅ All systems operational!",
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload, timeout=10)
    data = response.json()
    
    if data.get('ok'):
        print(f"   ✅ Message sent successfully!")
        print(f"   Message ID: {data.get('result', {}).get('message_id')}")
        print(f"\n🎉 CHECK YOUR TELEGRAM APP NOW!")
    else:
        print(f"   ❌ Failed to send message!")
        print(f"   Error: {data}")
        
        # Common errors
        error_desc = data.get('description', '')
        if 'chat not found' in error_desc.lower():
            print(f"\n💡 SOLUTION:")
            print(f"   Your CHAT_ID might be wrong.")
            print(f"   1. Open Telegram")
            print(f"   2. Start a chat with your bot")
            print(f"   3. Send any message to it")
            print(f"   4. Get your chat ID from @userinfobot")
        elif 'unauthorized' in error_desc.lower():
            print(f"\n💡 SOLUTION:")
            print(f"   Your BOT_TOKEN might be wrong.")
            print(f"   1. Go to @BotFather on Telegram")
            print(f"   2. Send /mybots")
            print(f"   3. Select your bot")
            print(f"   4. Get a new token")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

print("\n" + "="*60)
print("✅ TEST COMPLETE!")
print("="*60)
