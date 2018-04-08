// AxisState.cpp : 实现文件
//

#include "stdafx.h"
#include "CV_System.h"
#include "AxisState.h"
#include "afxdialogex.h"


// CAxisState 对话框

IMPLEMENT_DYNAMIC(CAxisState, CDialog)

CAxisState::CAxisState(CWnd* pParent /*=NULL*/)
	: CDialog(CAxisState::IDD, pParent)
{

}

CAxisState::~CAxisState()
{
}

void CAxisState::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CAxisState, CDialog)
END_MESSAGE_MAP()


// CAxisState 消息处理程序


void CAxisState::PostNcDestroy()
{
	// TODO: 在此添加专用代码和/或调用基类

	CDialog::PostNcDestroy();
	delete this;
}


void CAxisState::OnCancel()
{
	// TODO: 在此添加专用代码和/或调用基类

	//CDialog::OnCancel();
	DestroyWindow();
}
