// PM2 Ecosystem — JoSho Trader
// Start: pm2 start ecosystem.config.js
// Monitor: pm2 logs trading-autopilot
// Status: pm2 list
// NEVER restart during market hours (rate limit risk)

module.exports = {
  apps: [
    {
      name: "trading-autopilot",
      script: "autopilot.py",
      interpreter: "python",
      args: "",
      cwd: "C:/josho-trader",
      watch: false,
      autorestart: true,
      restart_delay: 60000,        // 60s between restarts (Groww rate limit)
      max_restarts: 3,             // max 3 restarts per window
      min_uptime: 30000,           // must run 30s to count as "started"
      kill_timeout: 5000,
      env: {
        PYTHONUNBUFFERED: "1",
      },
      error_file: "C:/josho-trader/logs/pm2-error.log",
      out_file: "C:/josho-trader/logs/pm2-out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
  ],
};
