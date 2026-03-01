use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Bar, BarChart, BarGroup, Block, Borders, Cell, Paragraph, Row, Table},
    Frame, Terminal,
};
use serde::Deserialize;

// ===== CLI Arguments =====

#[derive(Parser, Debug)]
#[command(name = "hybridqa-tui", about = "HybridQA-Net Terminal UI")]
struct Args {
    /// API base URL
    #[arg(long, default_value = "http://localhost:8000")]
    url: String,

    /// Username for auto-login
    #[arg(long, default_value = "admin")]
    username: String,

    /// Password for auto-login
    #[arg(long, default_value = "password123")]
    password: String,
}

// ===== API Types =====

#[derive(Debug, Deserialize, Clone)]
struct TokenResponse {
    access_token: String,
    #[allow(dead_code)]
    token_type: String,
    #[allow(dead_code)]
    expires_in: u64,
}

#[derive(Debug, Deserialize, Clone)]
struct DetectionBox {
    region_name: String,
    class_name: String,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    yolo_confidence: f64,
    clip_similarity: f64,
}

#[derive(Debug, Deserialize, Clone)]
struct GridRegion {
    region_name: String,
    clip_similarity: f64,
    #[allow(dead_code)]
    crop_width: u32,
    #[allow(dead_code)]
    crop_height: u32,
}

#[derive(Debug, Deserialize, Clone)]
struct SimilarityEntry {
    region_name: String,
    similarity: f64,
}

#[derive(Debug, Deserialize, Clone)]
struct YoloCLIPResponse {
    label: i32,
    verdict: String,
    confidence: f64,
    threshold: f64,
    yolo_detections: Vec<DetectionBox>,
    grid_regions: Vec<GridRegion>,
    best_region: String,
    best_similarity: f64,
    all_similarities: Vec<SimilarityEntry>,
    query: String,
    normalized_query: String,
    processing_time_s: f64,
    cached: bool,
}

// ===== App State =====

#[derive(Debug, Clone, Copy, PartialEq)]
enum FocusedField {
    ImagePath,
    Query,
    SubmitButton,
}

enum AppMessage {
    LoginSuccess(String),
    LoginError(String),
    AnalyzeSuccess(YoloCLIPResponse),
    AnalyzeError(String),
}

struct App {
    api_base: String,
    username: String,
    password: String,
    token: Option<String>,
    image_path: String,
    query: String,
    use_cache: bool,
    focused: FocusedField,
    status: String,
    login_status: String,
    result: Option<YoloCLIPResponse>,
    tx: Sender<AppMessage>,
    rx: Receiver<AppMessage>,
}

impl App {
    fn new(api_base: String, username: String, password: String) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            api_base,
            username,
            password,
            token: None,
            image_path: String::new(),
            query: String::new(),
            use_cache: true,
            focused: FocusedField::ImagePath,
            status: String::from("Нажмите 'a' для входа, 'q' для выхода"),
            login_status: String::new(),
            result: None,
            tx,
            rx,
        }
    }

    fn poll_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::LoginSuccess(token) => {
                    self.token = Some(token);
                    self.login_status = format!("✓ Вход выполнен как {}", self.username);
                    self.status = String::from("Tab: поле | Enter: анализ | q: выход");
                }
                AppMessage::LoginError(e) => {
                    self.login_status = format!("✗ Ошибка входа: {}", e);
                }
                AppMessage::AnalyzeSuccess(r) => {
                    self.status = format!(
                        "✓ Готово: {} | {:.2}s | best={} ({:.3})",
                        r.verdict, r.processing_time_s, r.best_region, r.best_similarity
                    );
                    self.result = Some(r);
                }
                AppMessage::AnalyzeError(e) => {
                    self.status = format!("✗ Ошибка: {}", e);
                }
            }
        }
    }

    fn do_login(&self) {
        let tx = self.tx.clone();
        let base = self.api_base.clone();
        let user = self.username.clone();
        let pass = self.password.clone();
        thread::spawn(move || {
            match login_request(&base, &user, &pass) {
                Ok(t) => tx.send(AppMessage::LoginSuccess(t)).ok(),
                Err(e) => tx.send(AppMessage::LoginError(e)).ok(),
            };
        });
    }

    fn do_analyze(&mut self) {
        let Some(token) = self.token.clone() else {
            self.status = String::from("Сначала войдите ('a')");
            return;
        };
        if self.image_path.is_empty() {
            self.status = String::from("Укажите путь к изображению");
            return;
        }
        self.status = String::from("Анализирую...");
        let tx = self.tx.clone();
        let base = self.api_base.clone();
        let path = self.image_path.clone();
        let query = self.query.clone();
        let use_cache = self.use_cache;
        thread::spawn(move || {
            match analyze_request(&base, &token, &path, &query, use_cache) {
                Ok(r) => tx.send(AppMessage::AnalyzeSuccess(r)).ok(),
                Err(e) => tx.send(AppMessage::AnalyzeError(e)).ok(),
            };
        });
    }

    fn next_focus(&mut self) {
        self.focused = match self.focused {
            FocusedField::ImagePath => FocusedField::Query,
            FocusedField::Query => FocusedField::SubmitButton,
            FocusedField::SubmitButton => FocusedField::ImagePath,
        };
    }

    fn handle_char(&mut self, c: char) {
        match self.focused {
            FocusedField::ImagePath => self.image_path.push(c),
            FocusedField::Query => self.query.push(c),
            FocusedField::SubmitButton => {}
        }
    }

    fn handle_backspace(&mut self) {
        match self.focused {
            FocusedField::ImagePath => { self.image_path.pop(); }
            FocusedField::Query => { self.query.pop(); }
            FocusedField::SubmitButton => {}
        }
    }
}

// ===== HTTP =====

fn login_request(base: &str, user: &str, pass: &str) -> Result<String, String> {
    let url = format!("{}/api/v1/auth/token", base);
    let body = serde_json::json!({"username": user, "password": pass});
    let client = reqwest::blocking::Client::new();
    let res = client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .send()
        .map_err(|e| e.to_string())?;
    if !res.status().is_success() {
        return Err(format!("HTTP {}", res.status()));
    }
    let t: TokenResponse = res.json().map_err(|e| e.to_string())?;
    Ok(t.access_token)
}

fn analyze_request(
    base: &str,
    token: &str,
    path: &str,
    query: &str,
    use_cache: bool,
) -> Result<YoloCLIPResponse, String> {
    let url = format!("{}/api/v1/analyze/v2", base);
    let file_bytes = std::fs::read(path).map_err(|e| format!("Файл: {}", e))?;
    let filename = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("image.jpg")
        .to_string();
    let mime = if filename.ends_with(".png") { "image/png" } else { "image/jpeg" };

    let part = reqwest::blocking::multipart::Part::bytes(file_bytes)
        .file_name(filename)
        .mime_str(mime)
        .map_err(|e| e.to_string())?;

    let form = reqwest::blocking::multipart::Form::new()
        .part("image", part)
        .text("query", query.to_string())
        .text("use_cache", if use_cache { "true" } else { "false" });

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;

    let res = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .multipart(form)
        .send()
        .map_err(|e| e.to_string())?;

    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().unwrap_or_default();
        return Err(format!("HTTP {}: {}", status, body));
    }

    res.json::<YoloCLIPResponse>().map_err(|e| e.to_string())
}

// ===== Drawing =====

const ZONE_ABBR: &[(&str, &str)] = &[
    ("top-left", "TL"),
    ("top-center", "TC"),
    ("top-right", "TR"),
    ("center-left", "CL"),
    ("center", "C"),
    ("center-right", "CR"),
    ("bottom-left", "BL"),
    ("bottom-center", "BC"),
    ("bottom-right", "BR"),
];

fn abbr(region: &str) -> &str {
    for (r, a) in ZONE_ABBR {
        if *r == region { return a; }
    }
    region
}

fn draw(f: &mut Frame, app: &App) {
    let full = f.area();

    // Top-level horizontal split: Left(40) | Right(fill)
    let h_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(40), Constraint::Fill(1)])
        .split(full);

    draw_left(f, app, h_chunks[0]);
    draw_right(f, app, h_chunks[1]);
}

fn draw_left(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // image path
            Constraint::Length(5),  // query
            Constraint::Length(3),  // submit
            Constraint::Length(3),  // status
            Constraint::Fill(1),    // auth hint
        ])
        .split(area);

    // Image path
    let ip_style = if app.focused == FocusedField::ImagePath {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    };
    let ip_block = Block::default()
        .borders(Borders::ALL)
        .title("Путь к файлу")
        .border_style(ip_style);
    let ip = Paragraph::new(app.image_path.as_str()).block(ip_block);
    f.render_widget(ip, chunks[0]);

    // Query
    let q_style = if app.focused == FocusedField::Query {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    };
    let q_block = Block::default()
        .borders(Borders::ALL)
        .title("Запрос")
        .border_style(q_style);
    let q = Paragraph::new(app.query.as_str())
        .block(q_block)
        .wrap(ratatui::widgets::Wrap { trim: false });
    f.render_widget(q, chunks[1]);

    // Submit button
    let sb_style = if app.focused == FocusedField::SubmitButton {
        Style::default().fg(Color::Black).bg(Color::Blue).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Blue)
    };
    let sb_text = if app.focused == FocusedField::SubmitButton {
        "[ Enter → Анализировать ]"
    } else {
        "  Tab → Анализировать  "
    };
    let sb = Paragraph::new(sb_text)
        .style(sb_style)
        .block(Block::default().borders(Borders::ALL).title("Отправить"));
    f.render_widget(sb, chunks[2]);

    // Status
    let status = Paragraph::new(app.status.as_str())
        .style(Style::default().fg(Color::Yellow))
        .block(Block::default().borders(Borders::ALL).title("Статус"));
    f.render_widget(status, chunks[3]);

    // Auth hint
    let hint_text = format!(
        "{}\n\na = войти как {}\nTab = следующее поле\nq/Esc = выход\nc = кэш: {}",
        &app.login_status,
        &app.username,
        if app.use_cache { "вкл" } else { "выкл" }
    );
    let hint = Paragraph::new(hint_text)
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL).title("Подсказки"));
    f.render_widget(hint, chunks[4]);
}

fn draw_right(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(11), // grid table (3×3 cells × ~3 rows + borders)
            Constraint::Length(14), // bar chart
            Constraint::Length(3),  // verdict
            Constraint::Fill(1),    // extra info
        ])
        .split(area);

    draw_grid_table(f, app, chunks[0]);
    draw_bar_chart(f, app, chunks[1]);
    draw_verdict(f, app, chunks[2]);
    draw_info(f, app, chunks[3]);
}

fn draw_grid_table(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title("Grid регионы (3×3)");

    // Grid order: TL, TC, TR, CL, C, CR, BL, BC, BR
    let grid_order = [
        "top-left", "top-center", "top-right",
        "center-left", "center", "center-right",
        "bottom-left", "bottom-center", "bottom-right",
    ];

    let make_cell = |region: &str, app: &App| -> Cell {
        let sim = app.result.as_ref().and_then(|r| {
            r.grid_regions.iter().find(|g| g.region_name == region).map(|g| g.clip_similarity)
        });
        let is_best = app.result.as_ref().map(|r| r.best_region == region).unwrap_or(false);
        let threshold = app.result.as_ref().map(|r| r.threshold).unwrap_or(0.25);

        let text = match sim {
            Some(s) => format!("{}\n{:.3}", abbr(region), s),
            None => format!("{}\n  —  ", abbr(region)),
        };

        let style = if is_best {
            Style::default().fg(Color::White).bg(Color::Red).add_modifier(Modifier::BOLD)
        } else if sim.map(|s| s >= threshold).unwrap_or(false) {
            Style::default().fg(Color::Black).bg(Color::Yellow)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        Cell::from(text).style(style)
    };

    let rows: Vec<Row> = (0..3).map(|row_i| {
        let cells: Vec<Cell> = (0..3).map(|col_i| {
            make_cell(grid_order[row_i * 3 + col_i], app)
        }).collect();
        Row::new(cells).height(3)
    }).collect();

    let widths = [
        Constraint::Ratio(1, 3),
        Constraint::Ratio(1, 3),
        Constraint::Ratio(1, 3),
    ];

    let table = Table::new(rows, widths).block(block);
    f.render_widget(table, area);
}

fn draw_bar_chart(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title("Схожесть по регионам");

    if let Some(result) = &app.result {
        let threshold = result.threshold;
        let best = &result.best_region;

        let mut sims = result.all_similarities.clone();
        sims.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        let bars: Vec<Bar> = sims.iter().map(|s| {
            let value = (s.similarity * 1000.0) as u64;
            let color = if s.region_name == *best {
                Color::Red
            } else if s.similarity >= threshold {
                Color::Yellow
            } else {
                Color::DarkGray
            };
            Bar::default()
                .value(value)
                .label(Line::from(abbr(&s.region_name)))
                .style(Style::default().fg(color))
                .value_style(Style::default().fg(Color::White).bg(color))
        }).collect();

        let chart = BarChart::default()
            .block(block)
            .bar_width(3)
            .bar_gap(1)
            .direction(Direction::Horizontal)
            .data(BarGroup::default().bars(&bars));
        f.render_widget(chart, area);
    } else {
        let empty = Paragraph::new("Нет данных для отображения")
            .style(Style::default().fg(Color::DarkGray))
            .block(block);
        f.render_widget(empty, area);
    }
}

fn draw_verdict(f: &mut Frame, app: &App, area: Rect) {
    if let Some(result) = &app.result {
        let (verdict_color, verdict_text) = if result.label == 1 {
            (Color::Green, format!("✓ PASS — {}", result.verdict))
        } else {
            (Color::Red, format!("✗ FAIL — {}", result.verdict))
        };
        let cached_str = if result.cached { " [кэш]" } else { "" };
        let text = format!(
            "{}{} | conf={:.3} | best={} ({:.3}) | {:.2}s",
            verdict_text, cached_str, result.confidence,
            result.best_region, result.best_similarity, result.processing_time_s
        );
        let p = Paragraph::new(text)
            .style(Style::default().fg(verdict_color).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL).title("Вердикт"));
        f.render_widget(p, area);
    } else {
        let p = Paragraph::new("Результат появится после анализа")
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL).title("Вердикт"));
        f.render_widget(p, area);
    }
}

fn draw_info(f: &mut Frame, app: &App, area: Rect) {
    if let Some(result) = &app.result {
        let mut lines = vec![
            Line::from(vec![
                Span::styled("Запрос: ", Style::default().fg(Color::DarkGray)),
                Span::raw(&result.query),
            ]),
            Line::from(vec![
                Span::styled("Норм.:  ", Style::default().fg(Color::DarkGray)),
                Span::styled(&result.normalized_query, Style::default().fg(Color::Cyan)),
            ]),
        ];

        if !result.yolo_detections.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("YOLO объектов: {}", result.yolo_detections.len()),
                Style::default().fg(Color::Yellow),
            )));
            for det in result.yolo_detections.iter().take(5) {
                lines.push(Line::from(Span::styled(
                    format!(
                        "  {} [{},{}]-[{},{}] conf={:.2} clip={:.3}",
                        det.class_name, det.x1, det.y1, det.x2, det.y2,
                        det.yolo_confidence, det.clip_similarity
                    ),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }

        let p = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Детали"))
            .wrap(ratatui::widgets::Wrap { trim: false });
        f.render_widget(p, area);
    } else {
        let help = Paragraph::new(
            "Как использовать:\n\
             1. Нажмите 'a' для автовхода\n\
             2. Tab → введите путь к изображению\n\
             3. Tab → введите запрос\n\
             4. Tab → Enter для анализа\n\
             'c' — переключить кэш"
        )
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL).title("Справка"));
        f.render_widget(help, area);
    }
}

// ===== Main =====

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(args.url, args.username, args.password);
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        app.poll_messages();
        terminal.draw(|f| draw(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::ZERO);

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('a') => app.do_login(),
                    KeyCode::Char('c') => {
                        app.use_cache = !app.use_cache;
                    }
                    KeyCode::Tab => app.next_focus(),
                    KeyCode::Enter => {
                        if app.focused == FocusedField::SubmitButton {
                            app.do_analyze();
                        }
                    }
                    KeyCode::Backspace => app.handle_backspace(),
                    KeyCode::Char(c) => app.handle_char(c),
                    _ => {}
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
