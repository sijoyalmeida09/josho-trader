// PM2 Ecosystem — JoSho Trader
//
// Usage:
//   pm2 start C:\Sijoy_2.0\automation\josho-trader\ecosystem.config.js
//   pm2 logs josho-trader
//   pm2 stop josho-trader
//   pm2 delete josho-trader
//
// NOTE: groww_bot.py reads secrets from vault.master.env via its own
// _load_vault_env() loader, so no env values are injected here. The bot
// must be able to find the vault file at:
//   C:\Sijoy_2.0\automation\secrets-os\vault.master.env
//
// The `--server` flag is added by the Integration agent in a later step to
// launch the Flask HTTP layer (bot_server.py) alongside the trading loop.
// Until that flag is wired, you can temporarily drop it from `args` below
// and the bot will still run with `--live` only.

module.exports = {
  apps: [
    {
      name: "josho-trader",
      script: "python",
      args: [
        "C:\\Sijoy_2.0\\automation\\josho-trader\\groww_bot.py",
        "--live",
        "--server",
      ],
      cwd: "C:\\Sijoy_2.0\\automation\\josho-trader",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      min_uptime: "60s",
      restart_delay: 5000,
      kill_timeout: 8000,
      env: {},
      out_file: "C:\\Sijoy_2.0\\automation\\josho-trader\\logs\\pm2-out.log",
      error_file: "C:\\Sijoy_2.0\\automation\\josho-trader\\logs\\pm2-error.log",
      time: true,
    },
    // ----------------------------------------------------------------------
    // Future worker: situational-awareness signals scanner (Layer 4).
    // Disabled until the signals worker module is implemented. To enable,
    // uncomment this block and add the module entry point under
    // automation/josho-trader/.
    // ----------------------------------------------------------------------
    // {
    //   name: "josho-trader-signals",
    //   script: "python",
    //   args: [
    //     "C:\\Sijoy_2.0\\automation\\josho-trader\\signals_worker.py",
    //     "--loop",
    //   ],
    //   cwd: "C:\\Sijoy_2.0\\automation\\josho-trader",
    //   interpreter: "none",
    //   autorestart: true,
    //   max_restarts: 10,
    //   min_uptime: "60s",
    //   restart_delay: 5000,
    //   kill_timeout: 8000,
    //   env: {},
    //   out_file: "C:\\Sijoy_2.0\\automation\\josho-trader\\logs\\signals-out.log",
    //   error_file: "C:\\Sijoy_2.0\\automation\\josho-trader\\logs\\signals-error.log",
    //   time: true,
    // },
  ],
};
