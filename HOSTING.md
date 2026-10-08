# JoSho Trader — Hosting Runbook

> **Goal:** keep the bot online 24/7, for **$0/month**, with a public IP that Groww accepts.
> **Today:** runs on Sijoy's Windows PC under PM2.
> **North star:** migrate to **Oracle Cloud Always-Free ARM VM** when ready.
>
> No deadline. Read sections 1 → 2 → 3 → 4 in order when you're about to migrate.
> Sections 5 and 6 are reference-only.

---

## 1. Current state — Windows PC

### What's wired today

- Bot: `python automation/josho-trader/groww_bot.py` under PM2 (`ecosystem.config.js`)
- Flask HTTP layer: `bot_server.py` bound to `http://127.0.0.1:8080`
- Public access: Cloudflare Tunnel via `automation/scripts/setup-trader-tunnel.ps1`
  → `trader.joshoit.com` → `127.0.0.1:8080`
- Notifier: Telegram (`SHARED__TELEGRAM_BOT_TOKEN` from vault)
- Database: Supabase (free tier)
- **Groww IP whitelist:** the PC's current public IP is already registered with Groww

### Pros

- Zero hosting cost
- Hardware already paid for
- IP already accepted by Groww (no re-registration friction)
- Local debugging is instant — direct file access, no SSH
- GPU + 24GB RAM headroom for future local LLM scoring

### Cons

- PC must stay awake **24/7**. Sleep = bot dies.
- Windows Update can reboot at 3am and break the streak.
- Power outage → bot down until manual restart.
- No remote `/api/status` access without the Cloudflare tunnel running.
- ISP can rotate the public IP → Groww rejects orders until you re-register.

### Sleep prevention (mandatory)

Run **once**, as Administrator PowerShell:

```powershell
# Never sleep while plugged in
powercfg /change standby-timeout-ac 0
# Never hibernate while plugged in
powercfg /change hibernate-timeout-ac 0
# Display can still sleep — that's fine, doesn't kill the bot
powercfg /change monitor-timeout-ac 15
# Disable hibernation file entirely (frees ~10GB of disk)
powercfg /hibernate off
```

**Revert** (if PC becomes a space heater):

```powershell
powercfg /change standby-timeout-ac 30
powercfg /change hibernate-timeout-ac 60
powercfg /hibernate on
```

Verify after running:

```powershell
powercfg /query SCHEME_CURRENT SUB_SLEEP | Select-String "Sleep after"
```

Expected: `Current AC Power Setting Index: 0x00000000`.

### Disable Windows automatic restarts

Settings → Windows Update → **Active hours** → set to `1:00 AM – 11:59 PM` (max range)
so updates can't reboot during market hours.

Plus block the reboot via `gpedit.msc`:
**Computer Configuration → Administrative Templates → Windows Components → Windows Update**
→ enable `No auto-restart with logged on users for scheduled automatic updates installations`.

### Remote access to /api/status

Already covered by `automation/scripts/setup-trader-tunnel.ps1` — it installs Cloudflare Tunnel as a
Windows service and points `trader.joshoit.com` at `127.0.0.1:8080`. Free, unlimited bandwidth,
auto-reconnects, auto-starts.

Alternative: **Tailscale** (free for personal use, up to 100 devices).
Install on the PC and your phone/iPad. Reach the PC at its Tailscale IP (e.g. `100.x.y.z:8080`)
from anywhere. Useful as a fallback if Cloudflare ever flakes. https://tailscale.com/download

---

## 2. Migrating to Oracle Cloud Always-Free ARM VM

> **Why this is the answer:** 24GB RAM, 4 vCPU, 200GB storage, **$0/month forever**, true static
> public IP, runs in an Oracle data center with 99.95% uptime. The bot uses <500MB RAM and
> <50MB/day egress — vastly within the free tier.

### 2.0 Oracle Always-Free spec (confirmed Q4 2026)

From https://www.oracle.com/cloud/free/ and https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm:

| Resource | Always-Free allocation |
|---|---|
| Compute (ARM Ampere A1) | **4 OCPUs + 24 GB RAM** total (split across up to 4 instances) |
| Compute (x86 AMD) | 2× VM.Standard.E2.1.Micro (1/8 OCPU + 1 GB RAM each) |
| Block storage | 200 GB total |
| Bandwidth | **10 TB egress / month** |
| Public IPs | 2 reserved (free, persistent) |
| Cost | **$0 forever** as long as you stay within these limits |

**Use the ARM A1 shape — not the x86 micro.** The ARM shape is the only one with enough RAM.

> **Account warning:** Oracle requires a credit card to verify identity. They will not charge it
> as long as your account stays in "Always Free" mode. If you accidentally provision a paid
> resource, you'll get billed. **Stick to A1 Flex + ≤200 GB storage and you're safe.**

### 2.1 Create the Oracle Cloud account

1. Go to https://www.oracle.com/cloud/free/
2. Click **Start for free**
3. Use a real address. Use the email tied to your card.
4. Pick a **home region** close to your trading market. For Groww (India), pick
   **Mumbai (ap-mumbai-1)** or **Hyderabad (ap-hyderabad-1)**. Latency to api.groww.in will be
   <30ms.
5. Verify card + phone. Account is usually ready in ~10 minutes.

> **Home region is permanent.** You cannot change it later without opening a new account.

### 2.2 Generate an SSH key (on your Windows PC)

```powershell
ssh-keygen -t ed25519 -C "sijoy@trader" -f $env:USERPROFILE\.ssh\oracle_trader
```

- Press Enter twice (no passphrase, or set one and add it to ssh-agent later).
- This creates `oracle_trader` (private — never share) and `oracle_trader.pub` (public — upload).

Copy the public key contents:

```powershell
Get-Content $env:USERPROFILE\.ssh\oracle_trader.pub | clip
```

### 2.3 Create the VM instance

In the Oracle Cloud Console (https://cloud.oracle.com):

1. **Menu → Compute → Instances → Create instance**
2. Name: `josho-trader`
3. **Image and shape:**
   - Image: **Canonical Ubuntu 22.04** (LTS, ARM build)
   - Shape: click **Change shape** → category **Ampere** → **VM.Standard.A1.Flex**
   - OCPUs: `4`, Memory: `24 GB` (max free)
4. **Networking:**
   - Create new VCN (default settings are fine)
   - Subnet: public
   - **Assign a public IPv4 address: Yes**
5. **SSH keys:** paste the public key from step 2.2
6. **Boot volume:** keep default 50GB (well under the 200GB limit)
7. Click **Create**

VM is up in ~2 minutes. Note the **ephemeral public IP** shown on the instance page.

### 2.4 Reserve the public IP (CRITICAL)

The IP you got in step 2.3 is **ephemeral** — Oracle can rotate it on stop/start. Convert it to
a reserved IP so it never changes.

1. On the instance page → **Attached VNICs** → click the primary VNIC
2. **IPv4 Addresses** → click the three-dot menu next to the public IP → **Edit**
3. Select **Reserved Public IP** → **Reserve new public IP** → give it a name (`trader-static`)
4. Save.

Now the IP is yours forever (until you delete the reservation). Write it down:
`__________________________` (fill in after creating)

Docs: https://docs.oracle.com/en-us/iaas/Content/Network/Tasks/managingpublicIPs.htm

### 2.5 SSH in

```powershell
ssh -i $env:USERPROFILE\.ssh\oracle_trader ubuntu@<your-reserved-ip>
```

First time you'll get a fingerprint prompt — type `yes`.

### 2.6 Install base dependencies

```bash
# Update OS
sudo apt update && sudo apt upgrade -y

# Python 3.12 + pip + venv (Ubuntu 22.04 ships 3.10; we want 3.12 for parity with dev box)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip git curl build-essential

# Node 20 (for PM2)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# PM2 globally
sudo npm install -g pm2

# Confirm versions
python3.12 --version    # Python 3.12.x
node --version          # v20.x
pm2 --version
```

### 2.7 Copy the bot to the VM

**Option A — rsync only what's needed (recommended, smaller):**

From your Windows PC (PowerShell with WSL or Git Bash):

```bash
rsync -avz -e "ssh -i ~/.ssh/oracle_trader" \
  /c/Sijoy_2.0/automation/josho-trader/ \
  ubuntu@<your-ip>:/home/ubuntu/josho-trader/

# Copy vault separately (it's the secrets — handle with care)
scp -i ~/.ssh/oracle_trader \
  /c/Sijoy_2.0/automation/secrets-os/vault.master.env \
  ubuntu@<your-ip>:/home/ubuntu/josho-trader/vault.master.env
chmod 600 /home/ubuntu/josho-trader/vault.master.env  # on the VM
```

**Option B — git clone (if you push the repo to a private remote):**

```bash
# On VM
git clone https://github.com/sijoyalmeida09/josho-trader.git ~/josho-trader
cd ~/Sijoy_2.0/automation/josho-trader
# Then scp the vault file separately — DO NOT commit it
```

### 2.8 Python virtualenv + deps

```bash
cd /home/ubuntu/josho-trader
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.9 Re-register the new IP with Groww (DO NOT SKIP)

This is **the most critical step** in the entire migration. Without it the bot will get
`403 / IP not whitelisted` on every order and silently fail.

1. From the VM, get the public IP one more time to confirm:

   ```bash
   curl -s ifconfig.me
   ```

   This must match the reserved IP from step 2.4.

2. Log in to Groww web → https://groww.in/user/profile → **API Access** (or **Trading API**)
3. Find the **Static IP** field (the IP currently bound is your home PC's)
4. **Replace** it with the Oracle reserved IP
5. **Save.** Groww re-issues the API key context against the new IP. Wait 5 minutes for
   propagation.

> The Groww API key itself (`SOHAYA__GROWW_TOTP_TOKEN` in vault) doesn't change — only the IP
> binding does. The vault file you copied still works.

Test:

```bash
cd /home/ubuntu/josho-trader
source .venv/bin/activate
python groww_bot.py --scan-once --no-trade
```

If you see scan output and no `IP not whitelisted` errors → success.

### 2.10 Network rules

**Oracle Security List** (VCN → Subnet → Security List → Default):

- **Ingress:** only TCP 22 (SSH) from `0.0.0.0/0`. Nothing else.
- **Egress:** allow all (default). The bot needs to reach `api.groww.in:443`, Supabase,
  Groq, Telegram, and Cloudflare.

**Do not expose port 8080.** The Flask layer stays bound to `127.0.0.1` and is reached either
via SSH tunnel or via Cloudflare Tunnel (see 2.12).

**Ubuntu's `iptables`/`ufw`:** Oracle Ubuntu images ship with iptables rules that block
everything except port 22. If you need to open ports inside the VM later, also touch the
Security List — *both* layers gate traffic.

### 2.11 PM2 startup + run

`ecosystem.config.js` was written for Windows paths. Adjust on the VM:

```bash
cd /home/ubuntu/josho-trader

# Quick sanity check — make sure paths in ecosystem.config.js use forward slashes and
# point to /home/ubuntu/josho-trader/.venv/bin/python instead of a Windows path.
# If it's still Windows-flavoured, edit it before running PM2.

pm2 start ecosystem.config.js
pm2 status                       # bot + bot_server should be "online"
pm2 logs --lines 50              # tail logs to confirm Telegram heartbeat fired

# Make PM2 survive reboots
pm2 startup systemd -u ubuntu --hp /home/ubuntu
# It prints a sudo command — run that command exactly as shown, then:
pm2 save
```

Test reboot:

```bash
sudo reboot
# Wait 60s, then SSH back in
pm2 status
```

Bot should be online. If not — re-run `pm2 startup` and re-`save`.

### 2.12 (Optional) Cloudflare Tunnel for /api/status

The existing setup script is Windows-only. The Linux equivalent:

```bash
# Install cloudflared
curl -L --output cloudflared.deb \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb

# Auth (opens a browser URL — copy/paste into your laptop browser)
cloudflared tunnel login

# Create tunnel (or reuse the existing josho-trader UUID from data/trader-tunnel.json)
cloudflared tunnel create josho-trader-vm

# Config at ~/.cloudflared/config.yml:
cat > ~/.cloudflared/config.yml <<EOF
tunnel: <new-uuid-or-existing>
credentials-file: /home/ubuntu/.cloudflared/<uuid>.json

ingress:
  - hostname: trader.joshoit.com
    service: http://127.0.0.1:8080
  - service: http_status:404
EOF

# Install as systemd service
sudo cloudflared service install
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
sudo systemctl status cloudflared
```

Then point the Hostinger DNS CNAME at the new tunnel UUID
(`<uuid>.cfargotunnel.com`) — Claude can do this via Hostinger MCP.

### 2.13 Migration validation checklist

- [ ] `curl ifconfig.me` from VM matches Groww-registered IP
- [ ] `python groww_bot.py --scan-once --no-trade` runs cleanly
- [ ] `pm2 status` shows bot online
- [ ] Telegram receives a heartbeat in the next hour
- [ ] Supabase `trader_state` has a recent `updated_at`
- [ ] `https://trader.joshoit.com/api/status` returns JSON (if tunnel set up)
- [ ] Reboot the VM, confirm PM2 brings everything back

Once green: shut down the Windows PC bot (`pm2 stop ecosystem.config.js`). Keep the PC code as
the dev/staging copy.

---

## 3. Cost calculator

### Oracle Always-Free reality check

- **Compute + RAM + storage + IP + bandwidth:** $0 forever, no time limit.
- **Bandwidth math:** 10 TB/month egress = ~333 GB/day. The trader bot does scan polls, order
  posts, and Telegram pings — call it **<100 MB/day in real life**. You'd need to run ~3300×
  more traffic to hit the cap.
- **Idle-reclaim rule:** Oracle reserves the right to reclaim Always-Free instances if "idle for
  7+ days" (their language). The bot's continuous Python loop + PM2 health probe = CPU is never
  idle → safe. (Docs: https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm#reclamation)

### Free-hosting alternatives

| Provider | Cost | Static IP | 24/7 | Verdict |
|---|---|---|---|---|
| **Oracle Always-Free A1** | **$0** | ✅ reserved | ✅ | **WIN — pick this** |
| Hostinger VPS (KVM 1) | ₹299/mo (~$3.60) | ✅ | ✅ | Skip — paid |
| AWS t2.micro free tier | $0 first **12 months only** | ✅ | ✅ | Skip — expires; then $8/mo |
| GitHub Actions | $0 | ❌ rotating | ❌ scheduled only | Skip — not realtime, IP rotates |
| Fly.io free | $0 (3× shared-cpu-1x micro) | ❌ shared IP, $2/mo for dedicated | ✅ | Maybe later — IP isn't truly static on free |
| Google Cloud e2-micro free | $0 (1 instance, US regions only) | ✅ | ✅ | Maybe later — US-only and tiny (1GB RAM) |

> **Why not the AWS free tier?** It's a 12-month promo, not "always-free." You'll get billed in
> month 13. Oracle is the only major cloud offering genuinely permanent free compute with
> enough RAM for our workload.

---

## 4. Disaster recovery — Groww IP mismatch

**Symptom:**

- Telegram alerts stop
- Bot logs show `403 Forbidden`, `IP not whitelisted`, or `authentication failed`
- `/api/status` returns positions stuck, no new trades

**Diagnosis:**

```bash
# SSH to whatever's running the bot (Windows PC or Oracle VM)
curl -s ifconfig.me
```

Compare to the IP shown in Groww profile → API Access page.

**Fix:**

1. Copy the current `ifconfig.me` output
2. Log into Groww web → API Access → **Update Static IP** → paste new IP → save
3. Wait 5 minutes
4. Restart the bot:
   - **Linux:** `pm2 restart josho-trader`
   - **Windows:** `pm2 restart josho-trader` (same command)
5. Check Telegram for the next heartbeat
6. Confirm an order succeeds (or `--scan-once` runs clean)

**Prevention on Windows PC:** ISPs sometimes rotate the public IP on modem reboot. Either
(a) pay for a static IP from your ISP, or (b) migrate to Oracle where the IP is reserved.

**Prevention on Oracle:** make sure you completed step 2.4 (reserve the IP). Ephemeral IPs
on Oracle can rotate if you stop/start the instance.

---

## 5. Health monitoring (free)

### Built-in (already wired)

- **Telegram on startup** — bot sends a "🟢 trader online" message via
  `SHARED__TELEGRAM_BOT_TOKEN`
- **Hourly heartbeat** — `groww_bot.py` posts wallet + position count every hour
- **Error alerts** — any unhandled exception sends a Telegram message before exit
- **PM2 auto-restart** — if Python crashes, PM2 respawns within seconds and a "restart" event
  is logged

If Telegram goes silent for >2 hours during market hours → something is wrong.

### Optional: Uptime Kuma (self-hosted, free)

Run on the same Oracle VM (negligible RAM):

```bash
# Quick Docker install
sudo apt install -y docker.io
sudo docker run -d --restart=always -p 3001:3001 \
  -v uptime-kuma:/app/data --name uptime-kuma louislam/uptime-kuma:1
```

Browse to `http://<vm-ip>:3001` (you'll need to open port 3001 in the Security List
temporarily, or SSH-tunnel: `ssh -L 3001:127.0.0.1:3001 ubuntu@<vm-ip>` then go to
http://localhost:3001).

Add a **Push monitor** for the bot. Uptime Kuma gives you a URL like
`https://your-kuma/api/push/abc123`. Add it to the bot's hourly heartbeat code — if Kuma stops
seeing pings, it Telegrams/emails you. Belt + suspenders.

Docs: https://github.com/louislam/uptime-kuma

### Optional: external uptime ping (zero infra)

- **UptimeRobot** (https://uptimerobot.com) — free for 50 monitors, 5-min interval. Point it at
  `https://trader.joshoit.com/api/status`. Email + Telegram alerts on failure.
- **Better Stack** (https://betterstack.com/uptime) — free for 10 monitors, 3-min interval.
  Cleaner UI, free phone-call escalation on the lowest paid plan if you ever upgrade.

---

## 6. Future revenue threshold — when to leave Oracle Free

Don't pay for hosting until the bot earns its rent. Suggested triggers (whichever hits first):

- **3 consecutive profitable months** (net of brokerage, taxes, slippage), OR
- **Portfolio > ₹50,000**, OR
- **Daily volume > ₹100,000 across F&O legs** (Oracle latency to api.groww.in might start to
  show)

When triggered, your options (all support the same `pm2 + Python` setup, just re-run section 2):

| Option | Cost | Why pick this |
|---|---|---|
| **Hostinger VPS KVM 1** (₹299/mo, Mumbai) | ~$3.60/mo | Lowest latency to Groww (sub-5ms), local support, MCP already wired |
| DigitalOcean Basic Droplet (Bangalore) | $6/mo | Better ecosystem, easy snapshots, IPv4 included |
| AWS Lightsail (Mumbai, 1 GB plan) | $5/mo | If you already have AWS muscle memory |

Migration playbook is **identical to section 2**: provision → ssh → install deps → rsync code →
`pm2 start` → **re-register IP with Groww** → cut over.

> The IP-re-registration step is always the make-or-break. Everything else is mechanical.

---

## Quick reference card

```text
WHERE BOT RUNS TODAY:    Windows PC, PM2-supervised
PUBLIC URL:              https://trader.joshoit.com (Cloudflare Tunnel)
LOCAL PORT:              127.0.0.1:8080  (NEVER expose publicly)
GROWW IP REGISTERED:     <current PC public IP — check ifconfig.me to confirm>
NORTH STAR HOST:         Oracle Cloud A1 Flex, ap-mumbai-1, 4 OCPU / 24 GB
NORTH STAR COST:         $0/forever
MIGRATION TIME:          ~90 minutes (most of it waiting for apt/pip)
CRITICAL STEP YOU MUST NOT SKIP:
    Update the static IP in Groww profile to match the Oracle reserved IP
    BEFORE running the bot in live mode. Otherwise every order = 403.
```

---

## Links

- Oracle Free Tier overview: https://www.oracle.com/cloud/free/
- Always-Free resource limits: https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm
- Reserve a public IP: https://docs.oracle.com/en-us/iaas/Content/Network/Tasks/managingpublicIPs.htm
- Cloudflare Tunnel install (Linux): https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
- Tailscale download: https://tailscale.com/download
- Uptime Kuma: https://github.com/louislam/uptime-kuma
- PM2 startup script docs: https://pm2.keymetrics.io/docs/usage/startup/
- Groww Trading API help: https://groww.in/help/trading-api
