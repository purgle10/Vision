
// CV_SystemView.cpp : CCV_SystemView 类的实现
//

#include "stdafx.h"
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
#ifndef SHARED_HANDLERS
#include "CV_System.h"
#endif

#include "AxisState.h"
#include "MainFrm.h"

#include "CV_SystemDoc.h"
#include "CV_SystemView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CCV_SystemView

IMPLEMENT_DYNCREATE(CCV_SystemView, CFormView)

BEGIN_MESSAGE_MAP(CCV_SystemView, CFormView)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()

	ON_WM_SIZE()
	ON_BN_CLICKED(IDC_START, &CCV_SystemView::OnBnClickedStart)
	ON_WM_TIMER()
	ON_BN_CLICKED(IDC_IMWAIT, &CCV_SystemView::OnBnClickedImwait)
	ON_WM_CLOSE()
	ON_COMMAND(IDM_AxisState, &CCV_SystemView::OnAxisstate)
END_MESSAGE_MAP()


// CCV_SystemView 构造/析构

CCV_SystemView::CCV_SystemView()
	: CFormView(CCV_SystemView::IDD)
{
	// TODO: 在此处添加构造代码
	pStc=NULL;
	pDC=NULL;
	capture=NULL;
	pAxisDlg=NULL;
}

CCV_SystemView::~CCV_SystemView()
{
}

void CCV_SystemView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
}

BOOL CCV_SystemView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: 在此处通过修改
	//  CREATESTRUCT cs 来修改窗口类或样式

	return CFormView::PreCreateWindow(cs);
}

void CCV_SystemView::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();
	GetParentFrame()->RecalcLayout();
	//m_nMapMode = -1;
	ResizeParentToFit();
	pStc=(CStatic *)GetDlgItem(IDC_IMSHOW);//IDC_VIEW为Picture控件ID
	pStc->GetClientRect(rect);//将CWind类客户区的坐标点传给矩形
	pDC=pStc->GetDC(); //得到Picture控件设备上下文
	hDC=pDC->GetSafeHdc(); //得到控件设备上下文的句柄 

}

void CCV_SystemView::OnRButtonUp(UINT /* nFlags */, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CCV_SystemView::OnContextMenu(CWnd* /* pWnd */, CPoint point)
{
#ifndef SHARED_HANDLERS
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
#endif
}


// CCV_SystemView 诊断

#ifdef _DEBUG
void CCV_SystemView::AssertValid() const
{
	CFormView::AssertValid();
}

void CCV_SystemView::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}

CCV_SystemDoc* CCV_SystemView::GetDocument() const // 非调试版本是内联的
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CCV_SystemDoc)));
	return (CCV_SystemDoc*)m_pDocument;
}
#endif //_DEBUG


// CCV_SystemView 消息处理程序

void CCV_SystemView::OnSize(UINT nType, int cx, int cy)
{
	CFormView::OnSize(nType, cx, cy);
	//MessageBox(_T("MAXIMIZED!"));
	// TODO: 在此处添加消息处理程序代码
	//if (nType==3)
	//{
	//	MessageBox(_T("MAXIMIZED!"));
	//}
	//if (nType==SIZE_MAXIMIZED)
	//{
	//	MessageBox(_T("MAXIMIZED!"));
	//}
	//if (::IsZoomed(m_hWnd))
	//{
	//	MessageBox(_T("MAXIMIZED!"));
	//}
	//ImageShow();
}


void CCV_SystemView::OnBnClickedStart()
{
	// TODO: 在此添加控件通知处理程序代码
	//MessageBox(_T("OK!"));
	ImageShow();
}

void CCV_SystemView::ImageShow()
{
	if (!capture)
	{
		capture=cvCaptureFromCAM(0);
	}
	if (!capture)
	{
		AfxMessageBox(_T("无法获得摄像头!"));
		return;
	}
	SetTimer(1, 25, NULL);
}

void CCV_SystemView::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	if (1==nIDEvent)
	{
		IplImage* img =0;
		img = cvQueryFrame(capture);

		CvvImage m_CvvImage;
		m_CvvImage.CopyOf(img, 1);
		m_CvvImage.DrawToHDC(hDC, &rect);
	}

	CFormView::OnTimer(nIDEvent);
}


void CCV_SystemView::OnBnClickedImwait()
{
	// TODO: 在此添加控件通知处理程序代码
	KillTimer(1);
}


void CCV_SystemView::OnClose()
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	CFormView::OnClose();
}


void CCV_SystemView::OnAxisstate()
{
	// TODO: 在此添加命令处理程序代码
	pAxisDlg = new CAxisState();
	pAxisDlg->Create(IDD_AxisState, this);
	pAxisDlg->ShowWindow(SW_SHOW);
}
