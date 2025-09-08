module.exports = {
  apps : [{
    name: 'generation',
    script: 'serve-gpu-2.py',
    interpreter: '/venv/trellis/bin/python',
    args: '--port 8093'
  }]
};
